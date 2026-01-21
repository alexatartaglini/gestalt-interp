from typing import Iterable, Tuple, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def patch_index_to_image_coords(
    patch_index: Union[int, Iterable[int]],
    patch_size: int,
    image_size: Tuple[int, int],
    one_indexed: bool = True,
) -> Tuple[int, int]:
    """
    Convert a patch index to image coordinates.
    """
    if isinstance(patch_index, int):
        patch_index = [patch_index]

    if one_indexed:  # Account for cls
        patch_index = [i - 1 for i in patch_index]

    W, _ = image_size
    grid_w = W // patch_size

    coords = [((i % grid_w) * patch_size, (i // grid_w) * patch_size) for i in patch_index]

    return coords


def patch_grid_labels_mode1(
    im: Image.Image,
    patch_size: int,
    perimeter_shape_frac_min: float = 0.02,  # perimeter inclusion threshold
    target_shape_frac_min: float = 0.20,
    interior_shape_frac_min: float = 0.98,   # "majority shape" definition (tweakable)
):
    """
    Returns (npy, npx, is_perimeter, is_interior_shape, shape_frac)
    where is_perimeter / is_interior_shape / shape_frac are 2D arrays over patch grid.
    """
    if im.mode != "1":
        raise ValueError(f"Expected mode '1', got {im.mode!r}")

    W, H = im.size
    if (W % patch_size) != 0 or (H % patch_size) != 0:
        raise ValueError(f"Image size {(W, H)} must be divisible by patch_size={patch_size}")

    arr = (np.array(im, dtype=np.uint8) > 0).astype(np.uint8)  # 0/1

    npx = W // patch_size
    npy = H // patch_size
    n = npx * npy

    patches = (
        arr.reshape(npy, patch_size, npx, patch_size)
        .transpose(0, 2, 1, 3)
        .reshape(n, patch_size * patch_size)
    )

    has_one  = patches.any(axis=1)
    has_zero = (~patches.astype(bool)).any(axis=1)

    shape_frac = patches.mean(axis=1)

    is_bg = has_zero & (~has_one)

    raw_perimeter = has_one & has_zero
    # perimeter candidates must also have enough shape pixels
    keep_perimeter = raw_perimeter & (shape_frac >= perimeter_shape_frac_min)
    target = raw_perimeter & (shape_frac >= target_shape_frac_min)  # use a different threshold for target tokens

    # interior shape: NOT perimeter, but majority (or threshold) shape
    interior_shape = (~raw_perimeter) & (shape_frac >= interior_shape_frac_min)

    # reshape to grid
    is_perimeter = keep_perimeter.reshape(npy, npx)
    is_target = target.reshape(npy, npx)
    is_interior_shape = interior_shape.reshape(npy, npx)
    is_bg = is_bg.reshape(npy, npx)
    shape_frac_grid = shape_frac.reshape(npy, npx)

    return npy, npx, is_perimeter, is_target, is_interior_shape, is_bg, shape_frac_grid


def translation_dirs_and_neighbors_for_patch_size(
    transformation: str,
    npx: int,
    npy: int,
    is_perim: np.ndarray,
    is_target: np.ndarray,
    is_interior: np.ndarray,
    shape_frac_grid: np.ndarray,
    is_bg: np.ndarray,
    is_curve: bool,
    target_tokens_only: bool = True,
    perimeter_tokens_only: bool = False,
    target_shape_frac_min: float = 0.20,
):
    def in_bounds(r, c):
        return 0 <= r < npy and 0 <= c < npx

    def token_index_1based(r, c):
        return r * npx + c + 1

    # get neighbor flags with safe boundary handling
    def get(mat, rr, cc):
        return bool(mat[rr, cc]) if in_bounds(rr, cc) else False
    
    dirs_for_ps = {}
    neighbor_tokens_ps = {}

    # Iterate over patches of interest
    for r in range(npy):
        for c in range(npx):
            Lb = get(is_bg, r, c - 1)
            Rb = get(is_bg, r, c + 1)
            Ub = get(is_bg, r - 1, c)
            Db = get(is_bg, r + 1, c)

            if is_interior[r, c] and (is_curve or (Lb or Rb or Ub or Db)):
                is_perim[r, c] = True
                is_interior[r, c] = False

                if shape_frac_grid[r, c] >= target_shape_frac_min:
                    is_target[r, c] = True

            if target_tokens_only and not is_target[r, c]:
                continue
            elif perimeter_tokens_only and not is_perim[r, c]:
                continue

            if transformation == "translation" or "translation" in transformation:
                vertical_needed = Ub or Db
                horizontal_needed = Lb or Rb

                valid = []
                if vertical_needed:
                    valid += ["vertical"]          # translate up/down
                if horizontal_needed:
                    valid += ["horizontal"]        # translate left/right
                if not (vertical_needed or horizontal_needed):
                    is_target[r, c] = False
                    continue
                
                dirs_for_ps[token_index_1based(r, c)] = valid
            else:
                dirs_for_ps[token_index_1based(r, c)] = ["clockwise", "counterclockwise"]

            neighbor_tokens_ps[token_index_1based(r, c)] = {
                token_index_1based(r, c-1): "L",
                token_index_1based(r, c+1): "R",
                token_index_1based(r-1, c): "U",
                token_index_1based(r+1, c): "D",
                token_index_1based(r-1, c-1): "UL",
                token_index_1based(r-1, c+1): "UR",
                token_index_1based(r+1, c-1): "DL",
                token_index_1based(r+1, c+1): "DR",
            }

            invalid_neighbor_tokens = []

            for t, coords in zip(
                neighbor_tokens_ps[token_index_1based(r, c)].keys(),
                [(r, c-1), (r, c+1), (r-1, c), (r+1, c), (r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)]
            ):
                token_location = neighbor_tokens_ps[token_index_1based(r, c)][t]
                try: 
                    if is_bg[*coords]:
                        token_type = "b"
                    elif is_interior[*coords]:
                        token_type = "i"
                    else:
                        token_type = "c"
                    neighbor_tokens_ps[token_index_1based(r, c)][t] = {"location": token_location, "content_type": token_type}
                except IndexError:
                    invalid_neighbor_tokens.append(t)

            for t in invalid_neighbor_tokens:
                neighbor_tokens_ps[token_index_1based(r, c)].pop(t)

    return dirs_for_ps, neighbor_tokens_ps, is_perim, is_target, is_interior


def highlight_tokens_in_image(
    image: Image.Image,
    patch_size: int,
    token_indices: Iterable[int],
    *,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: int = 200,
    has_cls_token: bool = False,
    outline: bool = False,
    outline_color: Optional[Tuple[int, int, int]] = None,
    outline_width: int = 2,
) -> Image.Image:
    """
    Return a copy of `image` with the specified patch tokens highlighted.

    Args:
        image: PIL image.
        patch_size: Patch size in pixels (e.g., 14, 16).
        token_indices: Iterable of token indices to highlight (ints).
            - If has_cls_token=False: indices are 0..(num_patches-1)
            - If has_cls_token=True: 0 is CLS, and patch tokens are 1..num_patches
        color: RGB highlight color.
        alpha: Transparency (0..255) for the highlight fill.
        has_cls_token: Whether token_indices includes CLS at index 0.
        outline: If True, draw a rectangle outline in addition to the fill.
        outline_color: RGB outline color (defaults to `color` if None).
        outline_width: Outline width in pixels.

    Notes:
        - Image dimensions must be divisible by patch_size (or we crop to the largest divisible region).
        - Tokens outside the valid patch-token range are ignored.
        - CLS token (if present and requested) is ignored since it doesn't map to an image region.
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be a positive integer")

    # Work on a copy; ensure RGBA for compositing.
    base = image.convert("RGBA")

    W, H = base.size
    grid_w = W // patch_size
    grid_h = H // patch_size

    if grid_w == 0 or grid_h == 0:
        raise ValueError(f"patch_size={patch_size} is larger than the image size {base.size}")

    # If not divisible, crop to the largest divisible region
    crop_w = grid_w * patch_size
    crop_h = grid_h * patch_size
    if (crop_w, crop_h) != (W, H):
        base = base.crop((0, 0, crop_w, crop_h))
        W, H = base.size

    num_patches = grid_w * grid_h

    # Prepare overlay.
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    fill_rgba = (color[0], color[1], color[2], int(alpha))
    if outline_color is None:
        outline_color = color
    outline_rgba = (outline_color[0], outline_color[1], outline_color[2], 255)

    # Normalize / filter indices.
    idxs = set(int(i - 1) for i in token_indices)

    # If CLS token is included, it has no spatial region. Ignore it.
    if has_cls_token and 0 in idxs:
        idxs.remove(0)

    # Convert token index -> patch index in [0, num_patches-1]
    # If has_cls_token, patch token t maps to patch_idx = t - 1
    for t in idxs:
        patch_idx = t - 1 if has_cls_token else t
        if patch_idx < 0 or patch_idx >= num_patches:
            continue

        c = patch_idx % grid_w
        r = patch_idx // grid_w

        x0 = c * patch_size
        y0 = r * patch_size
        x1 = x0 + patch_size
        y1 = y0 + patch_size

        # Fill
        draw.rectangle([x0, y0, x1, y1], fill=fill_rgba)

        # Optional outline (slightly inset so it doesn't spill)
        if outline:
            # PIL rectangle outline draws centered on edge; inset for cleaner look.
            inset = max(0, outline_width // 2)
            draw.rectangle(
                [x0 + inset, y0 + inset, x1 - inset, y1 - inset],
                outline=outline_rgba,
                width=outline_width,
            )

    # Composite and return.
    out = Image.alpha_composite(base, overlay)
    return out.convert(image.mode) if image.mode != "RGBA" else out

def token_imshow(
    image: Image.Image,
    patch_size: int,
    grid: bool = True,
    grid_color: str = "white",
    patch_numbers: bool = True,
    highlight_tokens: Iterable[Tuple[int, int]] = None,
    show_original_image: bool = False,
) -> None:
        '''
        :param grid: bool
        :param grid_color: "white", "black", "red", or a tuple (r, g, b) with values between 0 and 255
        :param patch_numbers: bool
        '''
        plt.close()
        original_image = image
        
        if highlight_tokens is not None:
            image = highlight_tokens_in_image(image, patch_size, highlight_tokens)
        if grid:
            if grid_color == "white":
                grid_color = (255, 255, 255)
            elif grid_color == "black":
                grid_color = (0, 0, 0)
            elif grid_color == "red":
                grid_color = (255, 0, 0)
            else:
                grid_color = grid_color
            
            if patch_numbers:
                grid_im = Image.open(f"utils/grids/grid{patch_size}num.png")
            else:
                grid_im = Image.open(f"utils/grids/grid{patch_size}.png")
            grid_data = grid_im.getdata()
            
            new_data = []
            for item in grid_data:
                if item[0] < 100 and item[1] < 100 and item[2] < 100:
                    new_data.append((0, 0, 0, 0))
                else:
                    new_data.append((*grid_color, 255))
                    
            grid_im.putdata(new_data)
            
            bg = image.copy()
            bg.paste(grid_im, (0, 0), grid_im)
            image = bg

        tick_params = {
            "axis": "both",
            "which": "both",
            "bottom": False,
            "top": False,
            "left": False,
            "labelbottom": False,
            "labelleft": False
        }

        if show_original_image:
            _, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(original_image)
            ax[1].imshow(image)

            ax[0].tick_params(**tick_params)
            ax[1].tick_params(**tick_params)
        else:
            plt.imshow(image)
            plt.tick_params(**tick_params)
        
        plt.show()