from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode

import numpy as np
import torch


def preprocess_segmentation(segmentation, processor):
    """Need to scale and pad segmentation map to match
    processed image
    """

    # Get Relevant Quantities from processor
    size = getattr(processor, "size", None) or processor.image_size
    crop_size = (
        getattr(processor, "crop_size", None)
        or getattr(processor, "crop", None)
        or size
    )

    if isinstance(size, dict):
        try:
            resize_h, resize_w = size["height"], size["width"]
        except:
            resize_h, resize_w = size["shortest_edge"], size["shortest_edge"]
    else:
        resize_h = resize_w = int(size)

    if isinstance(crop_size, dict):
        crop_h, crop_w = crop_size["height"], crop_size["width"]
    else:
        crop_h = crop_w = int(crop_size)

    # Mask to Image
    segmentation = torch.tensor(np.asarray(segmentation), dtype=torch.int64)

    seg_float = segmentation.unsqueeze(0).unsqueeze(0).to(torch.float32)  # [1,1,H,W]
    seg_resized = F.interpolate(seg_float, size=(resize_h, resize_w), mode="nearest")
    seg_resized = seg_resized.squeeze(0).squeeze(0).to(torch.int32)     # [H,W]

    # --- center crop by slicing ---
    h, w = seg_resized.shape
    top = max((h - crop_h) // 2, 0)
    left = max((w - crop_w) // 2, 0)
    segm_crop = seg_resized[top:top + crop_h, left:left + crop_w]

    return np.array(segm_crop, dtype=np.int32)


def extract_patch_labels(patch_size, segmentation_mask):
    # Turn segmentation mask into a list of
    # (batch_size, H/patch_size * W/patch_size) labels, one per patch
    if isinstance(segmentation_mask, np.ndarray):
        segmentation_mask = torch.from_numpy(segmentation_mask)
    batches = segmentation_mask.shape[0]
    height = segmentation_mask.shape[1]
    width = segmentation_mask.shape[2]
    num_patches_height, num_patches_width = height // patch_size, width // patch_size
    patches = (
        segmentation_mask.reshape(
            batches, num_patches_height, patch_size, num_patches_width, patch_size
        )
        .permute(0, 1, 3, 2, 4)
        .reshape(batches, num_patches_height, num_patches_width, patch_size * patch_size)
    )
    return torch.mode(patches, dim=-1).values.reshape(batches, -1)


def get_hidden_states(model, processor, images, layer):
    inputs = processor(images, return_tensors="pt")
    inputs = {k: v.cuda().to(dtype=torch.bfloat16) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad():  # Ensure no gradients are computed
        hidden_states = model(**inputs, output_hidden_states=True).hidden_states[layer+1]
    return hidden_states[:, 1:, :]