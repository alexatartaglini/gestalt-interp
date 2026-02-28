import sys
import torch

sys.path.insert(1, "TransformerLens")
from transformer_lens.HookedViT import HookedViT
from transformers import AutoProcessor, AutoImageProcessor
from transformers import logging
logging.set_verbosity_error()


pretrain_paths = {
    "imagenet": "google/vit-base-patch{patch_size}-224-in21k",
    "clip": "openai/clip-vit-base-patch{patch_size}",
    "dinov2": "facebook/dinov2-base",
    "scratch": None,
}
pretrain_processor_classes = {
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
        cache_dir=".",
        device="auto"
    ):
        '''
        :param pretrain: "imagenet", "clip", "scratch", "dinov2", 
        :param patch_size: 16, 32
        :param scratch_architecture: "imagenet", "dinov2", "clip"
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

    def run_with_cache(self, im, remove_batch_dim=False):
        im = self.preprocess(im).to(self.device)
        logits, cache = self.model.run_with_cache(im, remove_batch_dim=remove_batch_dim)
        return logits, cache
    
    import torch

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

    def run_with_head_ablation(self, im, head_str, remove_batch_dim=False):
        # Head str is "layer_head-idx"
        head_str = head_str.split("_")
        layer = int(head_str[0])
        head = int(head_str[1])
        im = self.preprocess(im).to(self.device)
        hook_name, hook = self.make_head_uniform_over_patches(layer, head)
        self.model.add_hook(hook_name, hook)
        logits, cache = self.model.run_with_cache(
                im,
                remove_batch_dim=remove_batch_dim
            )
        self.model.reset_hooks(including_permanent=True)

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