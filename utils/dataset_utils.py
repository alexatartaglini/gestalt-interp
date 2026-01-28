import glob
import os
import shutil
import copy
from typing import Union
import uuid
import itertools
import tqdm
import yaml

import torch
from torch.utils.data import Dataset
torch.set_grad_enabled(False)

from PIL import Image, ImageDraw
import numpy as np
from matplotlib.colors import to_rgb

from .shape_utils import draw_shape
from .token_utils import patch_grid_labels_mode1, translation_dirs_and_neighbors_for_patch_size, patch_index_to_image_coords


class ExperimentDataset(Dataset):
    def __init__(self, experiment_path: str):
        self.experiment_path = experiment_path
        self.cfg = yaml.safe_load(open(os.path.join(self.experiment_path, "cfg.yaml")))

        image_paths = [f for f in glob.glob(os.path.join(self.experiment_path, "*.png"))]
        image_paths = sorted(image_paths)

        if self.cfg["transformation"] == "translation":
            signs = [f.split("/")[-1].split(".")[0].split("_")[-1][0] for f in image_paths]
            sign_idx = signs.index("-")

            image_paths = list(reversed(image_paths[sign_idx:])) + image_paths[:sign_idx]

        self.image_paths = image_paths
        self.n_images = len(self.image_paths)

        self.steps = [int(f.split("/")[-1].split(".")[0].split("_")[-1]) for f in self.image_paths]

        self.setup()

    def setup(self):
        segmentation = Image.open(self.cfg["segmentation_path"]).convert("1")

        npy, npx, is_perim, _, is_interior, is_bg, shape_frac_grid = patch_grid_labels_mode1(
            segmentation,
            self.cfg["patch_size"],
        )
        is_curve = self.cfg["shape_type"] in ["curve", "closedcurve"]
        target_token = self.cfg["target_token"]

        def in_bounds(r, c):
            return 0 <= r < npy and 0 <= c < npx

        def token_index_1based(r, c):
            return r * npx + c + 1

        def token_coords(idx):
            return (idx - 1) // npx, (idx - 1) % npx

        # get neighbor flags with safe boundary handling
        def get(mat, rr, cc):
            return bool(mat[rr, cc]) if in_bounds(rr, cc) else False

        for r in range(npy):
            for c in range(npx):
                Lb = get(is_bg, r, c - 1)
                Rb = get(is_bg, r, c + 1)
                Ub = get(is_bg, r - 1, c)
                Db = get(is_bg, r + 1, c)

                if is_interior[r, c] and (is_curve or (Lb or Rb or Ub or Db)):
                    is_perim[r, c] = True
                    is_interior[r, c] = False

        H, W = is_perim.shape

        labels = np.empty((H, W), dtype=object)

        labels[is_bg] = "bg"          # default / fallback
        labels[is_interior] = "interior"
        labels[is_perim] = "perimeter"

        target_token_coords = token_coords(target_token)
        labels[target_token_coords] = "target"

        src_tokens = self.cfg["neighbor_tokens"].keys()
        rel_locations = ["L", "R", "U", "D", "UL", "UR", "DL", "DR"]
        dest_token_info_per_src = {}
        src_token_info = {}

        for src in src_tokens:
            r, c = token_coords(src)
            dest_token_coords = [(r, c-1), (r, c+1), (r-1, c), (r+1, c), (r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)]
            dest_tokens = []
            dest_rel_locations = []
            for d, l in zip(dest_token_coords, rel_locations):
                if in_bounds(*d):
                    dest_tokens.append(d)
                    dest_rel_locations.append(l)

            dest_tokens = [
                {
                    "dest_token": token_index_1based(*d),
                    "dest_token_relative_location": rl,
                    "dest_token_type": labels[d[0], d[1]],
                    "dest_token_shape_frac": shape_frac_grid[d[0], d[1]].item(),
                }
                for d, rl in zip(dest_tokens, dest_rel_locations)
            ]
            dest_tokens.append({
                "dest_token": 0,
                "dest_token_relative_location": "none",
                "dest_token_type": "cls",
                "dest_token_shape_frac": -1,
            })

            dest_token_info_per_src[src] = dest_tokens

            src_token_info[src] = {
                "src_token": src,
                "src_token_type": labels[r, c],
                "src_token_shape_frac": shape_frac_grid[r, c].item(),
            }

        self.src_token_info = src_token_info
        self.dest_token_info_per_src = dest_token_info_per_src

    def get_max_aligned_index(self):
        return self.steps.index(0)

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        return Image.open(self.image_paths[idx]).convert("RGB")

    def __str__(self):
        rep = f"""
        ExperimentDataset:
            experiment_path: {self.experiment_path}
            n_images: {self.n_images}
            cfg:\n"""
        for k, v in self.cfg.items():
            rep += f"\t\t{k}: {v}\n"
        return rep


class ContinuityDatasetConfig:
    """
    Stores axes of variation for a continuity dataset. 
    """
    def __init__(self, cfg: dict = {}):
        self.dataset_root = cfg.get("dataset_root", "datasets")  ## Path to the dataset root
        self.dataset_path = cfg.get("dataset_path", "default_dataset")  ## Path to the dataset
        self.full_path = os.path.join(self.dataset_root, self.dataset_path)  ## Full path to the dataset
        self.shape_types = cfg.get("shape_types", ["circle"])  ## List of shapes: "circle", "oval", "square", "rectangle", "blob", "closedcurve", "curve"
        self.shape_complexities = cfg.get("shape_complexities", [10])  ## List of complexities for blobs and curves (i.e. # points)
        self.shape_sizes = cfg.get("shape_sizes", [80, 115, 150])  ## List of sizes for shapes, lengths for curves
        self.colors = cfg.get("colors", ["red", "blue", "green", "yellow", "purple", "orange", "brown", "pink", "gray", "black", "white"])  ## Possible shape & background colors
        self.controls = cfg.get("controls", ["rotated", "diff-color", "diff-color-rotated", "scrambled"])  ## List of control types: "rotated", "diff_color", "diff_color_rotated", "scrambled"
        self.transformation = cfg.get("transformation", "translation")  ## Transformation type: "translation", "rotation"
        self.transformation_step_size = cfg.get("transformation_step_size", 1)  ## How much to shift the target token, 1 = 1px
        self.image_size = cfg.get("image_size", (224, 224))  ## Size of the image
        self.patch_sizes = cfg.get("patch_sizes", (14, 16, 32))
        self.n_color_combinations = cfg.get("n_color_combinations", 3)  ## Number of color combinations to use
        self.curve_width = cfg.get("curve_width", 6)
        self.seed = cfg.get("seed", 42)

        self.setup()

    def setup(self):
        if self.patch_sizes is None:
            raise ValueError("patch_sizes cannot be None")
        self.max_patch_size = max(self.patch_sizes)

        if self.shape_types is None:
            raise ValueError("shape_types cannot be None")
        self.shape_types = self.shape_types if isinstance(self.shape_types, list) else [self.shape_types]

        if self.shape_complexities is None:
            self.shape_complexities = [None]
        elif isinstance(self.shape_complexities, list):
            self.shape_complexities = self.shape_complexities
        else:
            self.shape_complexities = [self.shape_complexities]
        
        if self.shape_sizes is None:
            raise ValueError("shape_sizes cannot be None")
        self.shape_sizes = self.shape_sizes if isinstance(self.shape_sizes, list) else [self.shape_sizes]

        for shape_size in self.shape_sizes:
            assert self.image_size[0] > 2*self.max_patch_size + shape_size, \
                f"Image size {self.image_size} is too small for shape size {shape_size} and max patch size {self.max_patch_size}. Maximum shape size is {self.image_size[0] - 2*self.max_patch_size - 1}."
            assert self.image_size[1] > 2*self.max_patch_size + shape_size, \
                f"Image size {self.image_size} is too small for shape size {shape_size} and max patch size {self.max_patch_size}. Maximum shape size is {self.image_size[1] - 2*self.max_patch_size - 1}."

        if self.colors is None:
            raise ValueError("colors cannot be None")
        self.colors = self.colors if isinstance(self.colors, list) else [self.colors]

        if self.controls is None:
            self.controls = [None]
        elif isinstance(self.controls, list):
            self.controls = self.controls
        else:
            self.controls = [self.controls]

        if self.transformation is None:
            raise ValueError("transformation cannot be None")
        self.transformation = self.transformation if isinstance(self.transformation, list) else [self.transformation]
        assert self.transformation[0] in ["translation", "rotation"], "transformation must be one of: translation, rotation"


class ContinuityDataset(Dataset):
    def __init__(self, cfg: Union[dict, ContinuityDatasetConfig] = None, load_if_exists: bool = True):        
        if load_if_exists: 
            self.cfg = cfg
            if isinstance(cfg, dict):
                self.cfg = ContinuityDatasetConfig(self.cfg)

            exists = self.validate_exists()

            if exists:
                self.rng = np.random.RandomState(self.cfg.seed)
                self.setup()
            else:
                raise FileNotFoundError(f"Dataset {self.cfg.dataset_path} does not exist or is ill-formed. Please instantiate this dataset using `create_from_config`.")
        
        else:  # load_if_exists is False
            if cfg is None:
                print("WARNING: No config provided. Creating a dataset using the default ContinuityDatasetConfig.")
                self.cfg = ContinuityDatasetConfig()
            else:
                self.cfg = cfg
            
            if isinstance(cfg, dict):
                self.cfg = ContinuityDatasetConfig(self.cfg)

            if os.path.exists(self.cfg.full_path):
                shutil.rmtree(self.cfg.full_path)

            self.rng = np.random.RandomState(self.cfg.seed)
        
            self.create()
            self.setup()

    def validate_exists(self):
        experiment_dir = os.path.join(self.cfg.full_path, "experiments")
        scene_dir = os.path.join(self.cfg.full_path, "scenes")
        segmentation_dir = os.path.join(self.cfg.full_path, "segmentations")
        ttc_dir = os.path.join(self.cfg.full_path, "theoretical_tuning_curves")

        exists = os.path.exists(self.cfg.full_path) and \
            os.path.exists(os.path.join(self.cfg.full_path, "experiments")) and \
            os.path.exists(os.path.join(self.cfg.full_path, "scenes")) and \
            os.path.exists(os.path.join(self.cfg.full_path, "segmentations")) and \
            os.path.exists(os.path.join(self.cfg.full_path, "theoretical_tuning_curves")) and \
            os.path.exists(os.path.join(self.cfg.full_path, "cfg.yaml"))

        if exists:
            cfg = self.cfg
            exists = len(os.listdir(ttc_dir)) == len(cfg.patch_sizes) and \
                len(os.listdir(scene_dir)) == len(os.listdir(segmentation_dir))

            if exists:
                scenes = [f[:-4] for f in os.listdir(scene_dir)]
                experiments = list(os.listdir(experiment_dir))

                for e in experiments:
                    exists = e in scenes
                    if not exists:
                        break

                if exists:
                    experiments = list(glob.glob(os.path.join(experiment_dir, "*", "*")))
                    for e in experiments:
                        exists = os.path.exists(os.path.join(e, "cfg.yaml")) and \
                            len(glob.glob(os.path.join(e, "*.png"))) > 0
                        if not exists:
                            break
        
        return exists

    def setup(self):
        experiment_paths = [
            f for f in os.listdir(os.path.join(self.cfg.full_path, "experiments"))
        ]
        self.experiment_paths = sorted(experiment_paths)

        subexperiment_paths = list(glob.glob(os.path.join(self.cfg.full_path, "experiments", "*", "*")))
        self.subexperiment_paths = sorted(subexperiment_paths)

    def filter_by_patch_size(self, patch_size: int):
        self.experiment_paths = [
            e for e in self.experiment_paths if f"PATCHSIZE-{patch_size}" in e
        ]
        self.subexperiment_paths = [
            e for e in self.subexperiment_paths if f"PATCHSIZE-{patch_size}" in e
        ]

    def filter_by_control(self, control: str):
        self.experiment_paths = [
            e for e in self.experiment_paths if control in e
        ]
        self.subexperiment_paths = [
            e for e in self.subexperiment_paths if control in e
        ]

    @classmethod
    def create_from_config(cls, cfg: Union[dict, ContinuityDatasetConfig, str] = None, dataset_root: str = "datasets"):
        if isinstance(cfg, str):
            if "/" not in cfg:
                cfg = f"{dataset_root}/{cfg}/cfg.yaml"
            cfg = yaml.safe_load(open(cfg))
            cfg = ContinuityDatasetConfig(cfg)
        elif isinstance(cfg, dict):
            cfg = ContinuityDatasetConfig(cfg)

        return cls(cfg, load_if_exists=False)

    @classmethod
    def load_from_path(cls, dataset_path: str, dataset_root: str = "datasets"):
        if "/" not in dataset_path:
            dataset_path = f"{dataset_root}/{dataset_path}"

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset {dataset_path} does not exist. Please create it using `create_from_config({dataset_path})`.")
        
        cfg = ContinuityDatasetConfig(yaml.safe_load(open(os.path.join(dataset_path, "cfg.yaml"))))
        return cls(cfg, load_if_exists=True)

    def _build_transformation_directions_and_perimeter_tokens(
        self,
        im: Image.Image,
        perimeter_shape_frac_min: float = 0.10,
        target_shape_frac_min: float = 0.20,
        interior_shape_frac_min: float = 1.0,
        target_tokens_only: bool = True,
        perimeter_tokens_only: bool = False,
        is_curve: bool = False,
    ):
        """
        Returns:
        transformation_directions[patch_size][token_index] = ["vertical"/"horizontal", ...] (direction for translation transformation)

        token_index is 1-indexed in row-major order.
        """
        transformation_directions = {}
        perimeter_token_indices = {} 
        target_token_indices = {}
        neighbor_tokens = {}
        for ps in self.cfg.patch_sizes:
            npy, npx, is_perim, is_target, is_interior, is_bg, shape_frac_grid = patch_grid_labels_mode1(
                im,
                ps,
                perimeter_shape_frac_min=perimeter_shape_frac_min,
                target_shape_frac_min=target_shape_frac_min,
                interior_shape_frac_min=interior_shape_frac_min,
            )

            dirs_for_ps, neighbor_tokens_ps, is_perim, is_target, is_interior = translation_dirs_and_neighbors_for_patch_size(
                self.cfg.transformation[0],
                npx, 
                npy,
                is_perim,
                is_target,
                is_interior,
                shape_frac_grid,
                is_bg,
                is_curve,
                target_tokens_only=target_tokens_only,
                perimeter_tokens_only=perimeter_tokens_only,
                target_shape_frac_min=target_shape_frac_min,
            )

            transformation_directions[ps] = dirs_for_ps
            neighbor_tokens[ps] = neighbor_tokens_ps
            perimeter_token_indices[ps] = (
                np.flatnonzero(is_perim.reshape(-1)) + 1
            ).tolist()
            target_token_indices[ps] = (
                np.flatnonzero(is_target.reshape(-1)) + 1
            ).tolist()
        
        return transformation_directions, perimeter_token_indices, target_token_indices, neighbor_tokens

    def _create_segmentation(self, image, shape_color, bg_color):
        shape_mask = Image.new("1", image.size)
        shape_pixels = image.load()
        mask_pixels = shape_mask.load()
        for y in range(image.size[1]):
            for x in range(image.size[0]):
                if shape_pixels[x, y] == shape_color:
                    mask_pixels[x, y] = 1
                elif shape_pixels[x, y] == bg_color:
                    mask_pixels[x, y] = 0
        return shape_mask

    def _create_image_and_segmentation(self, shape_type, shape_complexity, shape_size, transformation):
        colors = self.cfg.colors

        # Choose colors
        bg_color, shape_color, control_color = self.rng.choice(colors, size=3, replace=False)
        scene_colors = (bg_color.item(), shape_color.item(), control_color.item())
        bg_color = self.color_name_to_rgb(bg_color)
        shape_color = self.color_name_to_rgb(shape_color)

        # Create initial visual scene
        image = Image.new(mode="RGB", size=(self.cfg.image_size[0], self.cfg.image_size[1]), color=bg_color)
        draw = ImageDraw.Draw(image)

        draw_shape(
            shape_type=shape_type,
            image=image,
            draw=draw,
            shape_color=shape_color,
            shape_complexity=shape_complexity,
            margin=self.cfg.max_patch_size,
            width=self.cfg.curve_width,
            shape_size=shape_size,
            rng=self.rng,
        )

        # Create a new binary segmentation image where pixels matching shape_color are 1, bg_color are 0
        segmentation = self._create_segmentation(image, shape_color, bg_color)

        # Create transformation directions and perimeter tokens
        is_curve = shape_type in ["curve", "closedcurve"]
        interior_shape_frac_min = 0.1 if is_curve else 1.0
        transformation_directions, perimeter_tokens, target_tokens, neighbor_tokens = self._build_transformation_directions_and_perimeter_tokens(
            segmentation,
            is_curve=is_curve,
            interior_shape_frac_min=interior_shape_frac_min,
        )

        return image, segmentation, perimeter_tokens, target_tokens, transformation_directions, neighbor_tokens, scene_colors

    def _create_theoretical_tuning_curve(self, ps):
        ss = self.cfg.transformation_step_size
        step_offset = list(range(-ps, 0, ss)) + [0] + list(range(0, ps+1, ss))[1:]
        tuning_curve = [1 - abs(step / ps) for step in step_offset]
        return np.array(tuning_curve)

    def _create_control_patch(self, target_patch, experiment_cfg):
        control = experiment_cfg["control"]
        control_color = experiment_cfg["control_color"]
        control_rotation = experiment_cfg["control_rotation"]

        if control_color is not None:
            shape_color = self.color_name_to_rgb(experiment_cfg["shape_color"])
            control_color = self.color_name_to_rgb(experiment_cfg["control_color"])

            target_patch_data = list(target_patch.get_flattened_data())
            new_target_patch_data = []

            for r, g, b in target_patch_data:
                if r == shape_color[0] and g == shape_color[1] and b == shape_color[2]:
                    new_target_patch_data.append(control_color)
                else:
                    new_target_patch_data.append((r, g, b))

            target_patch.putdata(new_target_patch_data)

        if control_rotation is not None:
            target_patch = target_patch.rotate(control_rotation)

        if control == "scrambled":
            arr = np.array(target_patch)
            h, w = arr.shape[:2]

            flat = arr.reshape(-1, arr.shape[-1])
            self.rng.shuffle(flat)

            target_patch = Image.fromarray(flat.reshape(h, w, -1))

        return target_patch

    def _create_experiment(self, image, experiment_cfg: dict):
        # Get target patch and prepare the base image
        bg_color = self.color_name_to_rgb(experiment_cfg["bg_color"])
        patch_size = experiment_cfg["patch_size"]
        target_token = experiment_cfg["target_token"]

        target_token_image_coords = patch_index_to_image_coords(
            target_token,
            patch_size,
            image.size,
            one_indexed=True,
        )[0]
        target_token_image_coords = (
            target_token_image_coords[0],
            target_token_image_coords[1],
            target_token_image_coords[0] + patch_size,
            target_token_image_coords[1] + patch_size,
        )

        base = image.copy().convert("RGB")
        bg_patch = Image.new("RGB", (patch_size, patch_size), bg_color)
        target_patch = base.crop(box=(target_token_image_coords))
        base.paste(bg_patch, box=(target_token_image_coords[0], target_token_image_coords[1]))  # Overwrite the target patch with a bg patch

        # If we are creating a control that augments the color, then augment the target patch
        if experiment_cfg["control"] is not None:
            target_patch = self._create_control_patch(target_patch, experiment_cfg)

        # Perform the transformation
        transformation = experiment_cfg["transformation"]
        transformation_direction = experiment_cfg["transformation_direction"]

        ss = self.cfg.transformation_step_size
        if transformation == "translation":
            step_offset = list(range(-patch_size, 0, ss)) + [0] + list(range(0, patch_size+1, ss))[1:]
            filename_width = len(str(patch_size)) + 2
        elif transformation == "rotation":
            if transformation_direction == "clockwise":
                step_offset = list(range(0, 360, ss))
            elif transformation_direction == "counterclockwise":
                step_offset = list(range(0, -360, -ss))
            filename_width = len(str(360)) + 2

        for s, step in enumerate(step_offset):
            image_filename = f"offset_{step:+0{filename_width}d}.png"
            step_image = base.copy()

            if transformation == "translation":
                if transformation_direction == "horizontal":
                    step_box = (target_token_image_coords[0] + step, target_token_image_coords[1])
                elif transformation_direction == "vertical":
                    step_box = (target_token_image_coords[0], target_token_image_coords[1] + step)
            elif transformation == "rotation":
                step_box = (target_token_image_coords[0], target_token_image_coords[1])
                target_patch = target_patch.rotate(step, fillcolor=bg_color)

            step_image.paste(target_patch, box=step_box)
            step_image.save(os.path.join(experiment_cfg["experiment_path"], image_filename))
    
    def create(self, experiment_id_digits: int = 10):
        os.makedirs(self.cfg.full_path, exist_ok=True)
        os.makedirs(os.path.join(self.cfg.full_path, "scenes"), exist_ok=True)
        os.makedirs(os.path.join(self.cfg.full_path, "segmentations"), exist_ok=True)
        os.makedirs(os.path.join(self.cfg.full_path, "experiments"), exist_ok=True)
        os.makedirs(os.path.join(self.cfg.full_path, "theoretical_tuning_curves"), exist_ok=True)

        yaml_cfg = copy.deepcopy(self.cfg.__dict__)
        yaml_cfg["image_size"] = list(yaml_cfg["image_size"])
        yaml_cfg["patch_sizes"] = list(yaml_cfg["patch_sizes"])
        yaml.dump(yaml_cfg, open(os.path.join(self.cfg.full_path, "cfg.yaml"), "w"))

        # Create theoretical tuning curves
        for patch_size in self.cfg.patch_sizes:
            tuning_curve = self._create_theoretical_tuning_curve(patch_size)
            np.save(os.path.join(self.cfg.full_path, "theoretical_tuning_curves", f"PATCHSIZE-{patch_size}.npy"), tuning_curve)

        # Create experiments
        shape_types = self.cfg.shape_types
        shape_complexities = self.cfg.shape_complexities
        shape_sizes = self.cfg.shape_sizes
        controls = self.cfg.controls
        transformation = self.cfg.transformation
        n_color_combinations = self.cfg.n_color_combinations

        experiment_params = itertools.product(shape_types, shape_complexities, shape_sizes, transformation)
        experiment_paths = []
        for _, params in tqdm.tqdm(enumerate(experiment_params), desc="Creating dataset from config"):
            for _ in range(n_color_combinations):
                eid = uuid.uuid4().hex[:experiment_id_digits]
                while eid in experiment_paths:
                    eid = uuid.uuid4().hex[:experiment_id_digits]

                experiment_path = os.path.join(self.cfg.full_path, "experiments", eid)
                os.makedirs(experiment_path, exist_ok=True)  # Create overall experiment directory

                image, segmentation, perimeter_tokens, target_tokens, transformation_directions, neighbor_tokens, scene_colors = self._create_image_and_segmentation(
                    *params,
                )
                
                experiment_config = {
                    "seed": self.cfg.seed,
                    "control": None,
                    "shape_type": params[0],
                    "shape_complexity": params[1] if params[0] in ["blob", "curve", "closedcurve"] else None,
                    "shape_size": params[2],
                    "transformation": params[3],
                    "transformation_direction": None,
                    "neighbor_tokens": None,
                    "bg_color": scene_colors[0],
                    "shape_color": scene_colors[1],
                    "control_color": None,
                    "control_rotation": None,
                    "target_token": None,
                    "patch_size": None,
                    "experiment_path": None,
                    "tuning_curve_path": None,
                    "experiment_id": eid,
                    "scene_path": os.path.join(self.cfg.full_path, "scenes", f"{eid}.png"),
                    "segmentation_path": os.path.join(self.cfg.full_path, "segmentations", f"{eid}.png"),
                    "perimeter_tokens": perimeter_tokens,
                    "target_tokens": target_tokens,
                }
                scene_path = os.path.join(self.cfg.full_path, "scenes", f"{eid}.png")
                image.save(scene_path)

                segmentation_path = os.path.join(self.cfg.full_path, "segmentations", f"{eid}.png")
                segmentation.save(segmentation_path)

                min_number_target_tokens = min(len(target_tokens[patch_size]) for patch_size in self.cfg.patch_sizes)

                # Axes of variation: patch_size, target token, control type
                for patch_size in self.cfg.patch_sizes:
                    target_tokens_subset = self.rng.choice(target_tokens[patch_size], size=min_number_target_tokens, replace=False)

                    for target_token in target_tokens_subset:
                        target_token = target_token.item()
                        base_experiment_path = os.path.join(experiment_path, f"base_PATCHSIZE-{patch_size}_TOKEN-{target_token}")
                        os.makedirs(base_experiment_path, exist_ok=True)
                        cfg = copy.deepcopy(experiment_config)
                        cfg["patch_size"] = patch_size
                        cfg["target_token"] = target_token
                        cfg["transformation_direction"] = self.rng.choice(transformation_directions[patch_size][target_token]).item()
                        cfg["neighbor_tokens"] = neighbor_tokens[patch_size][target_token]
                        cfg["experiment_path"] = base_experiment_path
                        cfg["tuning_curve_path"] = os.path.join(self.cfg.full_path, "theoretical_tuning_curves", f"PATCHSIZE-{patch_size}.npy")
                        cfg["perimeter_tokens"] = perimeter_tokens[patch_size]
                        cfg["target_tokens"] = target_tokens[patch_size]
                        yaml.dump(cfg, open(os.path.join(base_experiment_path, "cfg.yaml"), "w"))
                        
                        # TODO: create experiment images
                        self._create_experiment(image, cfg)

                        if controls:
                            for control in controls:
                                control_path = os.path.join(experiment_path, f"{control}_PATCHSIZE-{patch_size}_TOKEN-{target_token}")
                                os.makedirs(control_path, exist_ok=True)
                                cfg = copy.deepcopy(experiment_config)
                                cfg["patch_size"] = patch_size
                                cfg["target_token"] = target_token
                                cfg["transformation_direction"] = self.rng.choice(transformation_directions[patch_size][target_token]).item()
                                cfg["neighbor_tokens"] = neighbor_tokens[patch_size][target_token]
                                cfg["experiment_path"] = control_path
                                cfg["tuning_curve_path"] = os.path.join(self.cfg.full_path, "theoretical_tuning_curves", f"PATCHSIZE-{patch_size}.npy")
                                cfg["perimeter_tokens"] = perimeter_tokens[patch_size]
                                cfg["target_tokens"] = target_tokens[patch_size]
                                cfg["control"] = control
                                cfg["control_color"] = scene_colors[2] if "color" in control else None
                                cfg["control_rotation"] = self.rng.choice([90, 180, 270]).item() if "rotated" in control else None
                                yaml.dump(cfg, open(os.path.join(control_path, "cfg.yaml"), "w"))
                                
                                # TODO: create experiment images
                                self._create_experiment(image, cfg)

    def color_name_to_rgb(self, name):
        r, g, b = to_rgb(name)  # returns floats in [0,1]
        return tuple(int(255 * x) for x in (r, g, b))

    def __getitem__(self, idx):
        experiment_path = self.subexperiment_paths[idx]
        experiment = ExperimentDataset(experiment_path)
        return experiment

    def __len__(self):
        return len(self.subexperiment_paths)
