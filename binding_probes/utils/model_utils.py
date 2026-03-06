import sys
import torch
import re

sys.path.insert(1, "TransformerLens")
from transformer_lens.HookedViT import HookedViT
from transformer_lens.loading_from_pretrained import convert_vit_weights
from transformers import AutoProcessor, AutoImageProcessor, ViTForImageClassification
from transformers import logging
logging.set_verbosity_error()


pretrain_paths = {
    "mae": "facebook/vit-mae-base",  # patch size = 16
    "dino": "facebook/dino-vitb{patch_size}",
    "imagenet": "google/vit-base-patch{patch_size}-224-in21k",
    "clip": "openai/clip-vit-base-patch{patch_size}",
    "dinov2": "facebook/dinov2-base",
    "scratch": None,
}
pretrain_processor_classes = {
    "mae": AutoImageProcessor,
    "dino": AutoImageProcessor,
    "imagenet": AutoImageProcessor,
    "clip": AutoProcessor,
    "dinov2": AutoImageProcessor,
    "scratch": AutoImageProcessor,
}

class Model:
    '''Wrapper class for ViT + processor'''
    def __init__(
        self,
        pretrain="imagenet",
        patch_size=32,
        scratch_architecture="imagenet",
        cache_dir="./cache/",
        device="auto"
    ):
        '''
        :param pretrain: "mae", "dino", "imagenet", "clip", "scratch", "dinov2", 
        :param patch_size: 14, 16, 32
        :param scratch_architecture: "mae", "dino", "imagenet", "dinov2", "clip"
        '''
        if pretrain is None:
            pretrain = "scratch"
        elif pretrain not in pretrain_paths:
            raise ValueError(f"Pretrain {pretrain} not in {pretrain_paths.keys()}")
        
        self.pretrain = pretrain
        self.patch_size = patch_size
        self.cache_dir = cache_dir
        self.load_type = scratch_architecture if pretrain == "scratch" else pretrain

        pretrain_path = pretrain_paths[self.load_type].format(patch_size=patch_size)
        pretrain_processor_class = pretrain_processor_classes[pretrain]

        if self.load_type in ["dino", "mae"]:
            tl_path = pretrain_paths["imagenet"].format(patch_size=patch_size)

            hf_model = ViTForImageClassification.from_pretrained(pretrain_path)

            self.model = HookedViT.from_pretrained(
                tl_path,
                cache_dir=cache_dir,
                is_clip=pretrain == "clip",
            )

            state_dict = convert_vit_weights(hf_model, self.model.cfg)
            self.model.load_state_dict(state_dict, strict=False)

            del hf_model
        else:
            self.model = HookedViT.from_pretrained(
                pretrain_path,
                cache_dir=cache_dir,
                is_clip=pretrain == "clip",
            )
        self.image_processor = pretrain_processor_class.from_pretrained(
            pretrain_path,
            cache_dir=cache_dir
        )
        if self.pretrain == "scratch":
            config = self.model.cfg
            self.model = HookedViT(
                config
            )

        self.device = device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.to(device)
        
    def __call__(self, im):
        im = self.preprocess(im).to(self.device)
        return self.model(im)

    def get_model_info(self):
        return {
            "model_pretrain": self.pretrain,
            "model_patch_size": self.patch_size,
            "model_architecture_type": self.load_type,
        }

    def make_head_uniform_over_patches(self, layer_idx: int, head_idx: int,):
        """
        Returns a (hook_name, hook_fn) pair that overwrites one head's attention
        pattern to be uniform over patch tokens (keys).
        
        hook_pattern shape is [batch, n_heads, q, k].
        """
        hook_name = f"blocks.{layer_idx}.attn.hook_pattern"

        def hook_fn(pattern: torch.Tensor, hook):
            # pattern: [B, H, Q, K] after softmax
            B, H, Q, K = pattern.shape
            device = pattern.device
            dtype = pattern.dtype

            # Uniform attention over keys
            uniform = torch.full((B, Q, K), 1.0 / K, device=device, dtype=dtype)
            # Overwrite just that head
            pattern[:, head_idx, :, :] = uniform
            return pattern

        return hook_name, hook_fn

    def attend_to_cls_only(self, layer_idx: int, head_idx: int,):
        """
        Returns a (hook_name, hook_fn) pair that overwrites one head's attention
        pattern to only attend to cls.
        
        hook_pattern shape is [batch, n_heads, q, k].
        """
        hook_name = f"blocks.{layer_idx}.attn.hook_pattern"

        def hook_fn(pattern: torch.Tensor, hook):
            # pattern: [B, H, Q, K] after softmax
            B, H, Q, K = pattern.shape
            device = pattern.device
            dtype = pattern.dtype

             # Create a tensor of pure zeros
            cls_only = torch.zeros((B, Q, K), device=device, dtype=dtype)
            
            # Force 100% of the attention weight to the CLS token (Key index 0)
            cls_only[:, :, 0] = 1.0
            pattern[:, head_idx, :, :] = cls_only
            return pattern

        return hook_name, hook_fn


    def mean_ablate(self, layer_idx: int, head_idx: int, mean_cache: dict):
        """
        Returns a (hook_name, hook_fn) pair that overwrites one head's output
        to the mean over the dataset.
        
        hook_pattern shape is [batch, n_heads, q, k].
        """
        hook_name = f"blocks.{layer_idx}.attn.hook_z"
        head_mean = mean_cache[layer_idx][head_idx]

        def hook_fn(z: torch.Tensor, hook):
            z[:, :, head_idx, :] = head_mean
            return z
        return hook_name, hook_fn
    

    def get_mean_head_activations(self, im):
        """
        Computes the mean activation for every head across all positions.
        Returns a dictionary mapping layer index to a tensor of shape [n_heads, d_head]
        """
        im = self.preprocess(im).to(self.device)
        
        # Storage for our results
        mean_activations = {}

        def save_mean_hook(value, hook):
            # value shape: [batch, pos, head_index, d_head]
            # We mean over batch (0) and position (1)
            mean_activations[hook.layer()] = value.mean(dim=(0, 1)).detach().cpu()

        # Create hook filters for all attention 'z' (head output) hooks
        # hook_z is the value per head before the final O matrix projection

        pattern = re.compile(r"blocks\.\d+\.attn\.hook_z")
        matching_hooks = [name for name in self.model.hook_dict.keys() if pattern.match(name)]

        # 2. Create the list of (name, hook_fn) tuples
        fwd_hooks = [(name, save_mean_hook) for name in matching_hooks]   

        # Run the model with these hooks
        self.model.run_with_hooks(
            im,
            fwd_hooks=fwd_hooks
        )

        return mean_activations


    def run_with_head_ablation(self, im, head_strs, mean_acts, remove_batch_dim=False):
        im = self.preprocess(im).to(self.device)

        # Head str is "layer_head-idx"
        # Add hooks to ablate all heads
        for head_str in head_strs:
            head_str = head_str.split("_")
            layer = int(head_str[0])
            head = int(head_str[1])
            hook_name, hook = self.mean_ablate(layer, head, mean_acts)
            self.model.add_hook(hook_name, hook)

        logits, cache = self.model.run_with_cache(
                im,
                remove_batch_dim=remove_batch_dim
            )
        self.model.reset_hooks(including_permanent=True)

        return logits, cache
    
    def run_with_cache(self, im, remove_batch_dim=False):
        im = self.preprocess(im).to(self.device)
        logits, cache = self.model.run_with_cache(im, remove_batch_dim=remove_batch_dim)
        return logits, cache

    def to(self, device):
        self.model.to(device)
        self.device = device
        return
    
    def preprocess(self, im):
        if str(type(self.image_processor)) == "<class 'transformers.models.clip.processing_clip.CLIPProcessor'>":
            im = self.image_processor(images=im, return_tensors='pt')["pixel_values"]
        else:
            im = self.image_processor(im, return_tensors='pt')["pixel_values"]
            
        return im

    def eval(self):
        self.model.eval()
        return