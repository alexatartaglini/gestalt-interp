import glob
import os
import random
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

from .shape_utils import draw_shape, generate_blob_keypoints, generate_close_blob_keypoints, draw_blob
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
        
        if segmentation.size != (224, 224):  # TODO: make this dynamic
            segmentation = segmentation.resize((224, 224), Image.Resampling.LANCZOS)

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
        self.shape_types = cfg.get("shape_types", ["circle"])  ## List of shapes: "circle", "oval", "square", "rectangle", "blob", "close_blobs", "closedcurve", "curve"
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
            self.controls = None
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

            if os.path.exists(os.path.join(self.cfg.full_path, "experiments")):
                shutil.rmtree(os.path.join(self.cfg.full_path, "experiments"))
            if os.path.exists(os.path.join(self.cfg.full_path, "theoretical_tuning_curves")):
                shutil.rmtree(os.path.join(self.cfg.full_path, "theoretical_tuning_curves"))
            if os.path.exists(os.path.join(self.cfg.full_path, "cfg.yaml")):
                os.remove(os.path.join(self.cfg.full_path, "cfg.yaml"))

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

        return image, segmentation, scene_colors

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
        target_patch = base.crop(box=(target_token_image_coords))

        if bg_color is not None:  # Overwrite the target patch with a bg patch
            bg_patch = Image.new("RGB", (patch_size, patch_size), bg_color)
            base.paste(bg_patch, box=(target_token_image_coords[0], target_token_image_coords[1])) 

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
        else:
            raise ValueError(f"Invalid transformation: {transformation}")

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

        # Check if scene images already exist
        scene_images = glob.glob(os.path.join(self.cfg.full_path, "scenes", "*.png"))
        scenes_exist = len(scene_images) > 0

        if scenes_exist:
            desc = "Creating dataset from existing scenes"
            eids = [f.split("/")[-1].replace(".png", "") for f in scene_images]
            experiment_params = [("", "", "", transformation[0]) for _ in eids]
        else:
            desc = "Creating dataset from config"
            experiment_params = itertools.product(shape_types, shape_complexities, shape_sizes, transformation)
            experiment_params = list(experiment_params)
            eids = []
            for _ in range(len(experiment_params)*n_color_combinations):
                eid = uuid.uuid4().hex[:experiment_id_digits]
                while eid in eids:
                    eid = uuid.uuid4().hex[:experiment_id_digits]
                eids.append(eid)
        
        for p, params in tqdm.tqdm(enumerate(experiment_params), desc=desc):
            for n in range(n_color_combinations):
                eid = eids[p*n_color_combinations + n]

                experiment_path = os.path.join(self.cfg.full_path, "experiments", eid)
                os.makedirs(experiment_path, exist_ok=True)  # Create overall experiment directory

                if scenes_exist:  # Use existing scene and segmentation
                    image = Image.open(os.path.join(self.cfg.full_path, "scenes", f"{eid}.png"))
                    image = image.resize((self.cfg.image_size[0], self.cfg.image_size[1]), Image.Resampling.LANCZOS)
                    segmentation = Image.open(os.path.join(self.cfg.full_path, "segmentations", f"{eid}.png")).convert("1")
                    segmentation = segmentation.resize((self.cfg.image_size[0], self.cfg.image_size[1]), Image.Resampling.LANCZOS)
                    scene_colors = ("", "", "")
                else:  # Create new scene based on experiment parameters
                    image, segmentation, scene_colors = self._create_image_and_segmentation(
                        *params,
                    )
                    scene_path = os.path.join(self.cfg.full_path, "scenes", f"{eid}.png")
                    image.save(scene_path)

                    segmentation_path = os.path.join(self.cfg.full_path, "segmentations", f"{eid}.png")
                    segmentation.save(segmentation_path)

                # Create transformation directions and perimeter tokens
                is_curve = params[0] in ["curve", "closedcurve"]
                interior_shape_frac_min = 0.1 if is_curve else 1.0
                transformation_directions, perimeter_tokens, target_tokens, neighbor_tokens = self._build_transformation_directions_and_perimeter_tokens(
                    segmentation,
                    is_curve=is_curve,
                    interior_shape_frac_min=interior_shape_frac_min,
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
        if len(name) == 0:
            return None
        r, g, b = to_rgb(name)  # returns floats in [0,1]
        return tuple(int(255 * x) for x in (r, g, b))

    def __getitem__(self, idx):
        experiment_path = self.subexperiment_paths[idx]
        experiment = ExperimentDataset(experiment_path)
        return experiment

    def __len__(self):
        return len(self.subexperiment_paths)


class ScrambledObjectDatasetConfig:
    """
    Stores axes of variation for a scrambled object dataset. 
    """
    def __init__(self, cfg: dict = {}):
        self.dataset_root = cfg.get("dataset_root", "datasets")  ## Path to the dataset root
        self.dataset_path = cfg.get("dataset_path", "default_dataset")  ## Path to the dataset
        self.full_path = os.path.join(self.dataset_root, self.dataset_path)  ## Full path to the dataset
        self.n_objects = cfg.get("n_objects", 2)
        self.shape_types = cfg.get("shape_types", ["circle"])  ## List of shapes: "circle", "oval", "square", "rectangle", "blob", "close_blobs", "closedcurve", "curve"
        self.shape_complexities = cfg.get("shape_complexities", [10])  ## List of complexities for blobs and curves (i.e. # points)
        self.shape_sizes = cfg.get("shape_sizes", [80, 115, 150])  ## List of sizes for shapes, lengths for curves
        self.colors = cfg.get("colors", ["red", "blue", "green", "yellow", "purple", "orange", "brown", "pink", "gray", "black", "white"])  ## Possible shape & background colors
        self.scrambling_types = cfg.get("scrambling_types", ["location", "orientation"])  ## Scrambling type: "location", "orientation"
        self.image_size = cfg.get("image_size", (224, 224))  ## Size of the image
        self.patch_sizes = cfg.get("patch_sizes", (14, 16, 32))
        self.n_images = cfg.get("n_images", 1000)
        self.n_test_images = cfg.get("n_test_images", 1000)
        self.min_distance_px = cfg.get("min_distance_px", 0)
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

        if self.scrambling_types is None:
            raise ValueError("scrambling_type cannot be None")
        self.scrambling_types = self.scrambling_types if isinstance(self.scrambling_types, list) else [self.scrambling_types]
        for scrambling_type in self.scrambling_types:
            assert scrambling_type in ["location", "orientation"], "scrambling_type must be one of: location, orientation"


class ScrambledObjectDataset(Dataset):
    def __init__(self, cfg: Union[dict, ScrambledObjectDatasetConfig] = None):
        self.cfg = cfg
        if isinstance(cfg, dict):
            self.cfg = ScrambledObjectDatasetConfig(self.cfg)

        self.rng = np.random.RandomState(self.cfg.seed)

        self.create()

    @classmethod
    def from_config(cls, cfg: Union[dict, ScrambledObjectDatasetConfig, str] = None, dataset_root: str = "datasets"):
        if cfg is None:
            cfg = ScrambledObjectDatasetConfig()
        elif isinstance(cfg, str):
            if "/" not in cfg:
                cfg = f"{dataset_root}/{cfg}/cfg.yaml"
            cfg = yaml.safe_load(open(cfg))
            cfg = ScrambledObjectDatasetConfig(cfg)
        elif isinstance(cfg, dict):
            cfg = ScrambledObjectDatasetConfig(cfg)

        return cls(cfg)

    def _bbox_min_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        ya, xa = np.where(a)
        yb, xb = np.where(b)

        xa_min, xa_max = xa.min(), xa.max()
        ya_min, ya_max = ya.min(), ya.max()
        xb_min, xb_max = xb.min(), xb.max()
        yb_min, yb_max = yb.min(), yb.max()

        # horizontal gap
        if xb_min > xa_max:
            dx = xb_min - xa_max
        elif xa_min > xb_max:
            dx = xa_min - xb_max
        else:
            dx = 0  # overlap/projection overlap in x

        # vertical gap
        if yb_min > ya_max:
            dy = yb_min - ya_max
        elif ya_min > yb_max:
            dy = ya_min - yb_max
        else:
            dy = 0

        return (dx**2 + dy**2) ** 0.5
    
    def _segmentations_overlap(
        self,
        seg1: Image.Image,
        seg2: Image.Image,
        min_distance_px: int = 0,
    ) -> bool:
        """
        Returns True if segmentations violate constraints:
        - overlap
        - OR are closer than min_distance_px

        seg1, seg2: PIL Images in mode "1"
        min_distance_px: minimum allowed pixel distance between objects
        """
        a = np.array(seg1, dtype=np.uint8)
        b = np.array(seg2, dtype=np.uint8)

        # 1. Hard overlap check
        if np.logical_and(a, b).any():
            return True

        # 2. Distance constraint (optional)
        if min_distance_px > 0:
            # Distance from every pixel to nearest pixel of object 1
            #dist_to_a = distance_transform_edt(~a)
            # Find minimum distance from object 2 to object 1
            #min_dist = dist_to_a[b].min()

            bbox_dist = self._bbox_min_distance(a, b)

            if bbox_dist < min_distance_px:
                return True

        return False

    def _get_quadrant_bounds(self, quadrant):
        W, H = self.cfg.image_size
        mx = W // 2
        my = H // 2
        margin = self.cfg.max_patch_size

        if quadrant == "tl":
            return ((margin, mx - margin), (margin, my - margin))
        if quadrant == "tr":
            return ((mx + margin, W - margin), (margin, my - margin))
        if quadrant == "bl":
            return ((margin, mx - margin), (my + margin, H - margin))
        if quadrant == "br":
            return ((mx + margin, W - margin), (my + margin, H - margin))

        raise ValueError(f"Unknown quadrant: {quadrant}")

    def _get_object_patch_indices(
        self,
        im: Image.Image,
        patch_size: int,
    ):
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

        is_bg = has_zero & (~has_one)
        is_shape = has_one

        is_bg = (np.flatnonzero(is_bg.reshape(-1)) + 1).tolist()
        is_shape = (np.flatnonzero(is_shape.reshape(-1)) + 1).tolist()

        return npx, npy, is_bg, is_shape

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

    def _finalize_two_object_image(
        self, image, seg1, seg2,
        aux_color, shape_color,
        bg_color, bg_color_name, shape_color_name
    ):
        # recolor aux → shape
        pixels = image.load()
        W, H = image.size
        for y in range(H):
            for x in range(W):
                if pixels[x, y] == aux_color:
                    pixels[x, y] = shape_color

        seg_all = self._create_segmentation(image, shape_color, bg_color)
        scene_colors = (bg_color_name, shape_color_name)

        return image, seg1, seg2, seg_all, scene_colors

    def _sample_blob_center(self, keypoints, center, center_bounds=None):
        W, H = self.cfg.image_size
        margin = self.cfg.max_patch_size

        rel_x = [x - center[0] for x, _ in keypoints]
        rel_y = [y - center[1] for _, y in keypoints]

        x_lo = margin - min(rel_x)
        x_hi = (W - margin) - max(rel_x)
        y_lo = margin - min(rel_y)
        y_hi = (H - margin) - max(rel_y)

        if center_bounds is not None:
            (bx_lo, bx_hi), (by_lo, by_hi) = center_bounds
            x_lo = max(x_lo, bx_lo)
            x_hi = min(x_hi, bx_hi)
            y_lo = max(y_lo, by_lo)
            y_hi = min(y_hi, by_hi)

        if x_lo >= x_hi or y_lo >= y_hi:
            raise ValueError("No valid center available for translated blob.")

        return (
            float(self.rng.uniform(x_lo, x_hi)),
            float(self.rng.uniform(y_lo, y_hi)),
        )

    def _draw_translated_blob(self, draw, keypoints, source_center, target_center, color):
        dx = target_center[0] - source_center[0]
        dy = target_center[1] - source_center[1]
        translated_keypoints = [(x + dx, y + dy) for x, y in keypoints]
        draw_blob(draw, translated_keypoints, fill=color, outline=color, samples_per_seg=30)

    def _sample_close_blob_offset(self, seg: Image.Image):
        arr = np.array(seg, dtype=np.uint8)
        ys, xs = np.where(arr > 0)
        if len(xs) == 0:
            raise ValueError("Empty segmentation for close blob offset sampling.")

        W, H = self.cfg.image_size
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        gap_px = int(self.rng.randint(1, 5))  # 1..4 empty pixels between bboxes

        candidates = []
        # Side-by-side placement is feasible only if the combined bbox can fit.
        if 2 * width + gap_px <= W:
            shift = float(width + gap_px)
            candidates.extend([(shift, 0.0), (-shift, 0.0)])
        if 2 * height + gap_px <= H:
            shift = float(height + gap_px)
            candidates.extend([(0.0, shift), (0.0, -shift)])

        if not candidates:
            return None

        idx = int(self.rng.randint(0, len(candidates)))
        return candidates[idx]

    def _sample_pair_anchor_delta(
        self,
        keypoints1,
        keypoints2,
        edge_bias: bool = True,
    ):
        W, H = self.cfg.image_size
        all_pts = keypoints1 + keypoints2
        xs = np.array([p[0] for p in all_pts], dtype=float)
        ys = np.array([p[1] for p in all_pts], dtype=float)
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()

        lo_dx = -min_x
        hi_dx = (W - 1) - max_x
        lo_dy = -min_y
        hi_dy = (H - 1) - max_y
        if lo_dx > hi_dx or lo_dy > hi_dy:
            return None

        if not edge_bias:
            dx = float(self.rng.uniform(lo_dx, hi_dx))
            dy = float(self.rng.uniform(lo_dy, hi_dy))
            return dx, dy

        jitter = float(self.rng.uniform(0.0, 2.5))
        if self.rng.uniform() < 0.5:
            dx = lo_dx + jitter
        else:
            dx = hi_dx - jitter
        if self.rng.uniform() < 0.5:
            dy = lo_dy + jitter
        else:
            dy = hi_dy - jitter

        dx = float(np.clip(dx, lo_dx, hi_dx))
        dy = float(np.clip(dy, lo_dy, hi_dy))
        return dx, dy

    def _create_image_and_segmentations(self, shape_type, shape_complexity, shape_size):
        colors = self.cfg.colors

        # --- choose colors ---
        bg_color_name, shape_color_name = self.rng.choice(colors, size=2, replace=False)
        bg_color_name = bg_color_name.item() if hasattr(bg_color_name, "item") else bg_color_name
        shape_color_name = shape_color_name.item() if hasattr(shape_color_name, "item") else shape_color_name

        remaining = [c for c in colors if c not in [bg_color_name, shape_color_name]]
        if not remaining:
            raise ValueError("Need ≥3 colors for two-object generation.")
        aux_color_name = self.rng.choice(remaining)
        aux_color_name = aux_color_name.item() if hasattr(aux_color_name, "item") else aux_color_name

        bg_color   = self.color_name_to_rgb(bg_color_name)
        shape_color = self.color_name_to_rgb(shape_color_name)
        aux_color   = self.color_name_to_rgb(aux_color_name)

        W, H = self.cfg.image_size
        max_outer_tries = 50
        max_inner_tries = 50

        # ------------------------------------------------------------
        # Phase 1: unconstrained resampling of object 1
        # ------------------------------------------------------------
        for _ in range(max_outer_tries):
            image = Image.new("RGB", (W, H), bg_color)
            draw = ImageDraw.Draw(image)

            blob_center = None
            blob_keypoints = None
            if shape_type == "blob":
                blob_center, blob_keypoints = generate_blob_keypoints(
                    image_size=(W, H),
                    shape_size=shape_size,
                    shape_complexity=shape_complexity,
                    margin=self.cfg.max_patch_size,
                    rng=self.rng,
                    fill_fraction=0.99,
                )
                draw_blob(draw, blob_keypoints, fill=shape_color, outline=shape_color, samples_per_seg=30)
            elif shape_type == "close_blobs":
                blob_center, blob_keypoints = generate_close_blob_keypoints(
                    image_size=(W, H),
                    shape_size=shape_size,
                    shape_complexity=shape_complexity,
                    margin=0,
                    rng=self.rng,
                    fill_fraction=0.998,
                )
                draw_blob(draw, blob_keypoints, fill=shape_color, outline=shape_color, samples_per_seg=30)
            else:
                draw_shape(
                    shape_type, image, draw, shape_color,
                    shape_complexity, margin=self.cfg.max_patch_size,
                    width=self.cfg.curve_width, shape_size=shape_size, rng=self.rng
                )
            seg1 = self._create_segmentation(image, shape_color, bg_color)
            image_obj1 = image.copy()

            for _ in range(max_inner_tries):
                image = image_obj1.copy()
                draw = ImageDraw.Draw(image)

                if shape_type == "blob":
                    blob2_center = self._sample_blob_center(blob_keypoints, blob_center)
                    self._draw_translated_blob(draw, blob_keypoints, blob_center, blob2_center, aux_color)
                elif shape_type == "close_blobs":
                    offset = self._sample_close_blob_offset(seg1)
                    if offset is None:
                        continue
                    x_off, y_off = offset
                    blob2_keypoints = [(x + x_off, y + y_off) for x, y in blob_keypoints]
                    edge_bias = shape_size >= 0.35 * min(W, H)
                    delta = self._sample_pair_anchor_delta(blob_keypoints, blob2_keypoints, edge_bias=edge_bias)
                    if delta is None:
                        continue
                    dx, dy = delta
                    blob1_keypoints = [(x + dx, y + dy) for x, y in blob_keypoints]
                    blob2_keypoints = [(x + dx, y + dy) for x, y in blob2_keypoints]

                    image = Image.new("RGB", (W, H), bg_color)
                    draw = ImageDraw.Draw(image)
                    draw_blob(draw, blob1_keypoints, fill=shape_color, outline=shape_color, samples_per_seg=30)
                    draw_blob(draw, blob2_keypoints, fill=aux_color, outline=aux_color, samples_per_seg=30)
                    seg1 = self._create_segmentation(image, shape_color, bg_color)
                else:
                    draw_shape(
                        shape_type, image, draw, aux_color,
                        shape_complexity, margin=self.cfg.max_patch_size,
                        width=self.cfg.curve_width, shape_size=shape_size, rng=self.rng
                    )
                seg2 = self._create_segmentation(image, aux_color, bg_color)

                if shape_type == "close_blobs":
                    bbox_dist = self._bbox_min_distance(np.array(seg1, dtype=np.uint8), np.array(seg2, dtype=np.uint8))
                    # At least ~1 px apart and no more than a few pixels apart.
                    if (not self._segmentations_overlap(seg1, seg2, 0)) and 2.0 <= bbox_dist <= 5.0:
                        return self._finalize_two_object_image(
                            image, seg1, seg2, aux_color, shape_color,
                            bg_color, bg_color_name, shape_color_name
                        )
                elif not self._segmentations_overlap(seg1, seg2, self.cfg.min_distance_px):
                    return self._finalize_two_object_image(
                        image, seg1, seg2, aux_color, shape_color,
                        bg_color, bg_color_name, shape_color_name
                    )

        # ------------------------------------------------------------
        # Phase 2: quadrant-constrained fallback
        # ------------------------------------------------------------
        quadrants = ["tl", "tr", "bl", "br"]
        self.rng.shuffle(quadrants)

        quad = self.rng.choice(quadrants)
        bounds = self._get_quadrant_bounds(quad)

        image = Image.new("RGB", (W, H), bg_color)
        draw = ImageDraw.Draw(image)

        blob_center = None
        blob_keypoints = None
        if shape_type == "blob":
            blob_center, blob_keypoints = generate_blob_keypoints(
                image_size=(W, H),
                shape_size=shape_size,
                shape_complexity=shape_complexity,
                margin=self.cfg.max_patch_size,
                rng=self.rng,
                fill_fraction=0.99,
                center_bounds=bounds,
            )
            draw_blob(draw, blob_keypoints, fill=shape_color, outline=shape_color, samples_per_seg=30)
        elif shape_type == "close_blobs":
            blob_center, blob_keypoints = generate_close_blob_keypoints(
                image_size=(W, H),
                shape_size=shape_size,
                shape_complexity=shape_complexity,
                margin=0,
                rng=self.rng,
                fill_fraction=0.998,
            )
            draw_blob(draw, blob_keypoints, fill=shape_color, outline=shape_color, samples_per_seg=30)
        else:
            draw_shape(
                shape_type, image, draw, shape_color,
                shape_complexity, margin=self.cfg.max_patch_size,
                width=self.cfg.curve_width, shape_size=shape_size, rng=self.rng,
                center_bounds=bounds
            )
        seg1 = self._create_segmentation(image, shape_color, bg_color)
        image_obj1 = image.copy()

        for _ in range(max_inner_tries):
            image = image_obj1.copy()
            draw = ImageDraw.Draw(image)

            if shape_type == "blob":
                blob2_center = self._sample_blob_center(blob_keypoints, blob_center, center_bounds=bounds)
                self._draw_translated_blob(draw, blob_keypoints, blob_center, blob2_center, aux_color)
            elif shape_type == "close_blobs":
                offset = self._sample_close_blob_offset(seg1)
                if offset is None:
                    continue
                x_off, y_off = offset
                blob2_keypoints = [(x + x_off, y + y_off) for x, y in blob_keypoints]
                edge_bias = True
                delta = self._sample_pair_anchor_delta(blob_keypoints, blob2_keypoints, edge_bias=edge_bias)
                if delta is None:
                    continue
                dx, dy = delta
                blob1_keypoints = [(x + dx, y + dy) for x, y in blob_keypoints]
                blob2_keypoints = [(x + dx, y + dy) for x, y in blob2_keypoints]

                image = Image.new("RGB", (W, H), bg_color)
                draw = ImageDraw.Draw(image)
                draw_blob(draw, blob1_keypoints, fill=shape_color, outline=shape_color, samples_per_seg=30)
                draw_blob(draw, blob2_keypoints, fill=aux_color, outline=aux_color, samples_per_seg=30)
                seg1 = self._create_segmentation(image, shape_color, bg_color)
            else:
                draw_shape(
                    shape_type, image, draw, aux_color,
                    shape_complexity, margin=self.cfg.max_patch_size,
                    width=self.cfg.curve_width, shape_size=shape_size, rng=self.rng,
                    center_bounds=bounds
                )
            seg2 = self._create_segmentation(image, aux_color, bg_color)

            if shape_type == "close_blobs":
                bbox_dist = self._bbox_min_distance(np.array(seg1, dtype=np.uint8), np.array(seg2, dtype=np.uint8))
                if (not self._segmentations_overlap(seg1, seg2, 0)) and 2.0 <= bbox_dist <= 5.0:
                    return self._finalize_two_object_image(
                        image, seg1, seg2, aux_color, shape_color,
                        bg_color, bg_color_name, shape_color_name
                    )
            elif not self._segmentations_overlap(seg1, seg2, self.cfg.min_distance_px):
                return self._finalize_two_object_image(
                    image, seg1, seg2, aux_color, shape_color,
                    bg_color, bg_color_name, shape_color_name
                )

        # ------------------------------------------------------------
        # Hard failure (real config issue)
        # ------------------------------------------------------------
        raise RuntimeError(
            "Could not place two non-overlapping objects after resampling "
            "and quadrant-constrained fallback.\n"
            "Try increasing image_size, decreasing shape_size, or increasing patch margin."
        )

    def _create_scrambled_image(self, image, scrambling_type, cfg):
        """
        Create a scrambled image where each object is scrambled within itself only.

        - For 'location': patches of each object are shuffled among that object's patch set.
        - For 'orientation': each patch in each object patch set is independently rotated.
        """
        patch_size = cfg["patch_size"]
        object1_patch_indices = cfg["object1_patch_indices"]
        object2_patch_indices = cfg["object2_patch_indices"]

        if object1_patch_indices is None or object2_patch_indices is None:
            raise ValueError(
                "Expected cfg to contain 'object1_patch_indices' and 'object2_patch_indices'. "
                "Make sure you re-generated the dataset with the updated create() logic."
            )

        scrambled_image = image.copy()

        def scramble_patch_group(im, patch_indices):
            # map patch index -> (x0, y0)
            patch_to_image_coords = {}
            patches = []

            for patch_index in patch_indices:
                patch_image_coords = patch_index_to_image_coords(
                    patch_index,
                    patch_size,
                    im.size,
                    one_indexed=True,
                )[0]
                patch_to_image_coords[patch_index] = patch_image_coords

                x0, y0 = patch_image_coords
                patch_box = (x0, y0, x0 + patch_size, y0 + patch_size)
                patch = im.crop(box=patch_box)
                patches.append(patch)

            # scramble
            if scrambling_type == "location":
                random.shuffle(patches)
            elif scrambling_type == "orientation":
                for i, patch in enumerate(patches):
                    rotation = self.rng.choice([90, 180, 270])
                    patches[i] = patch.rotate(rotation)
            else:
                raise ValueError(f"Unknown scrambling_type: {scrambling_type}")

            # paste back into the scrambled_image only within this group
            for patch_index, patch in zip(patch_indices, patches):
                x0, y0 = patch_to_image_coords[patch_index]
                scrambled_image.paste(patch, box=(x0, y0))

        # scramble each object independently
        scramble_patch_group(image, object1_patch_indices)
        scramble_patch_group(image, object2_patch_indices)

        return scrambled_image
    
    def create(self, experiment_id_digits: int = 10):
        os.makedirs(self.cfg.full_path, exist_ok=True)

        yaml_cfg = copy.deepcopy(self.cfg.__dict__)
        yaml_cfg["image_size"] = list(yaml_cfg["image_size"])
        yaml_cfg["patch_sizes"] = list(yaml_cfg["patch_sizes"])
        yaml.dump(yaml_cfg, open(os.path.join(self.cfg.full_path, "cfg.yaml"), "w"))

        # Create experiments
        shape_types = self.cfg.shape_types
        shape_complexities = self.cfg.shape_complexities
        shape_sizes = self.cfg.shape_sizes
        scrambling_types = self.cfg.scrambling_types
        n_images = self.cfg.n_images

        experiment_paths = []
        for i in tqdm.tqdm(range(n_images), desc="Creating dataset", total=n_images):
            params = (
                random.choice(shape_types),
                random.choice(shape_complexities),
                random.choice(shape_sizes),
            )
            eid = uuid.uuid4().hex[:experiment_id_digits]
            while eid in experiment_paths:
                eid = uuid.uuid4().hex[:experiment_id_digits]

            experiment_path = os.path.join(self.cfg.full_path, eid)
            os.makedirs(experiment_path, exist_ok=True)  # Create overall experiment directory

            # NEW: two-object image + separate segmentations
            image, seg1, seg2, seg_all, scene_colors = self._create_image_and_segmentations(
                *params,
            )

            experiment_config = {
                "seed": self.cfg.seed,
                "shape_type": params[0],
                "shape_complexity": params[1] if params[0] in ["blob", "close_blobs", "curve", "closedcurve"] else None,
                "shape_size": params[2],
                "bg_color": scene_colors[0],
                "shape_color": scene_colors[1],
                "experiment_id": eid,
                "num_patches_x": None,
                "num_patches_y": None,
                "object_patch_indices": None,   # union of both objects
                "object1_patch_indices": None,  # new
                "object2_patch_indices": None,  # new
                "bg_patch_indices": None,
                "patch_size": None,
            }
            image.save(os.path.join(experiment_path, "base.png"))

            # Axes of variation: patch_size, target token, control type
            for patch_size in self.cfg.patch_sizes:
                # union segmentation: which patches contain any object pixels?
                npx, npy, bg_union, shape_union = self._get_object_patch_indices(seg_all, patch_size)
                # per-object segmentations: which patches belong to obj1 vs obj2?
                _, _, _, obj1_patches = self._get_object_patch_indices(seg1, patch_size)
                _, _, _, obj2_patches = self._get_object_patch_indices(seg2, patch_size)

                cfg = copy.deepcopy(experiment_config)
                cfg["num_patches_x"] = npx
                cfg["num_patches_y"] = npy
                cfg["object_patch_indices"] = shape_union          # union
                cfg["object1_patch_indices"] = obj1_patches        # obj1 only
                cfg["object2_patch_indices"] = obj2_patches        # obj2 only
                cfg["bg_patch_indices"] = bg_union
                cfg["patch_size"] = patch_size

                yaml.dump(
                    cfg,
                    open(os.path.join(experiment_path, f"cfg_PATCHSIZE-{patch_size}.yaml"), "w")
                )

                if i < self.cfg.n_test_images:  # Create scrambled for test only
                    for scrambling_type in scrambling_types:
                        scrambled_image_path = os.path.join(
                            experiment_path,
                            f"{scrambling_type}_PATCHSIZE-{patch_size}.png"
                        )
                        scrambled_image = self._create_scrambled_image(image, scrambling_type, cfg)
                        scrambled_image.save(scrambled_image_path)

    def color_name_to_rgb(self, name):
        r, g, b = to_rgb(name)  # returns floats in [0,1]
        return tuple(int(255 * x) for x in (r, g, b))

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass
