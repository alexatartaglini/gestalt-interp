import sys
import torch

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
    "dinov2": "facebook/dinov2-large",
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
        cache_dir="/data/alexart/models",
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