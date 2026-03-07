from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode

import numpy as np
import torch

import math
from collections import defaultdict
from typing import Optional, Tuple, Dict, Any

import math
from typing import Optional, Tuple

def _infer_grid_size(N: int, grid_size: Optional[Tuple[int,int]] = None) -> Tuple[int,int]:
    if grid_size is not None:
        H, W = grid_size
        if H * W != N:
            raise ValueError(f"grid_size {grid_size} incompatible with N={N}")
        return H, W
    side = int(round(math.sqrt(N)))
    if side * side != N:
        raise ValueError(f"Cannot infer square grid from N={N}; pass grid_size.")
    return side, side

def compute_l1_binned_balanced_mask(
    labels: torch.Tensor,
    grid_size: Optional[Tuple[int,int]] = None,
    *,
    num_bins: int = 8,
    max_pairs_per_bin: int = 4096,
    exclude_self: bool = True,
    seed: Optional[int] = None,
) -> torch.BoolTensor:
    """
    Returns mask [B,N,N] selecting equal-count positives and negatives per L1-distance bin.
    Fast: uses vectorized ops + sampling from indices.
    """
    if labels.ndim != 2:
        raise ValueError("labels must be (B,N)")

    B, N = labels.shape
    H, W = _infer_grid_size(N, grid_size)
    device = labels.device

    idx = torch.arange(N, device=device)
    r = idx // W
    c = idx % W

    dr = (r[:, None] - r[None, :]).abs()
    dc = (c[:, None] - c[None, :]).abs()
    l1 = dr + dc  # [N,N]

    # bin edges (0..max)
    max_l1 = int(l1.max().item())
    edges = torch.linspace(0, max_l1 + 1e-6, steps=num_bins + 1, device=device)

    out = torch.zeros((B, N, N), dtype=torch.bool, device=device)

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)

    eye = torch.eye(N, dtype=torch.bool, device=device)

    for b in range(B):
        lb = labels[b]
        fg = lb != 0
        cand = fg[:, None] & fg[None, :]
        if exclude_self:
            cand &= ~eye

        same = cand & (lb[:, None] == lb[None, :])
        diff = cand & (lb[:, None] != lb[None, :])

        # for each bin, sample min(#pos,#neg,max_pairs_per_bin) from each
        for k in range(num_bins):
            in_bin = (l1 >= edges[k]) & (l1 < edges[k+1])

            pos_idx = torch.nonzero(same & in_bin, as_tuple=False)  # [P,2]
            neg_idx = torch.nonzero(diff & in_bin, as_tuple=False)  # [Q,2]
            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue

            m = min(pos_idx.shape[0], neg_idx.shape[0], max_pairs_per_bin)
            if m <= 0:
                continue

            # sample without replacement
            pi = torch.randperm(pos_idx.shape[0], generator=gen, device=device)[:m]
            ni = torch.randperm(neg_idx.shape[0], generator=gen, device=device)[:m]

            out[b, pos_idx[pi,0], pos_idx[pi,1]] = True
            out[b, neg_idx[ni,0], neg_idx[ni,1]] = True

    return out


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


def cfg2labels(file):
    obj1_patches = file["object1_patch_indices"]
    obj2_patches = file["object2_patch_indices"]

    bg_patches = file["bg_patch_indices"]

    total_patches = np.max(obj1_patches + obj2_patches + bg_patches)

    # Ensure no overlapping patches!
    filtered_obj1_patches = list(set(obj1_patches) - set(obj2_patches))
    filtered_obj2_patches = list(set(obj2_patches) - set(obj1_patches))

    # We get rid of the [CLS] token, so 0 index these object patches
    obj1_patches = [idx - 1 for idx in filtered_obj1_patches] 
    obj2_patches = [idx - 1 for idx in filtered_obj2_patches]

    obj_mask = np.zeros(total_patches)
    obj_mask[obj1_patches] = 1 # mask segmenting object 1 from background
    obj_mask[obj2_patches] = 2 # mask segmenting object 2 from background

    return torch.Tensor(obj_mask)


def img2states(img, processor, model):
    inputs = processor(img, return_tensors="pt")
    inputs = {k: v.cuda().to(dtype=torch.bfloat16) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad():  # Ensure no gradients are computed
        hidden_states = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states
    return hidden_states # Always only one image here