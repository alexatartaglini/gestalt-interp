import argparse
import os
import json
import yaml
from collections import defaultdict

from PIL import Image

import numpy as np

import torch
from transformers import AutoImageProcessor, AutoModel, ViTModel, ViTConfig

from tqdm import tqdm
from utils import processing

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        default="openai/clip-vit-base-patch32",
        help="Model to use as backbone for analysis",
    )

    parser.add_argument(
        "-dataset",
        default="./cache/scrambled_blobs",
        type=str,
    )

    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    parser.add_argument(
        "-outfolder", default="./cache", type=str, help="Directory to store results"
    )

    parser.add_argument(
        "-random_seed",
        default=19,
        type=int
    )

    args = parser.parse_args()
    return args


def process_dataset(args, split, processor, model):

    patch_size = model.config.patch_size
    model.eval()

    dataset = os.path.join(args.dataset, split)

    data = os.listdir(dataset)

    labels = []
    hidden_states_dict = defaultdict(list)

    for datum in tqdm(data):
        if datum == "cfg.yaml":
            continue

        img_path = os.path.join(dataset, datum, "base.png")
        cfg_path = os.path.join(dataset, datum, f"cfg_PATCHSIZE-{patch_size}.yaml")

        labels.append(processing.cfg2labels(yaml.safe_load(open(cfg_path, "r"))))
        all_states = processing.img2states(Image.open(img_path).convert("RGB"), processor, model)

        for layer in args.layers:
            hidden_states = all_states[layer+1] # +1 skips embedding layer
            hidden_states_dict[layer].append(hidden_states[0, 1:, ])

    for k in hidden_states_dict.keys():
        layer_states = hidden_states_dict[k]
        layer_states = torch.stack(layer_states, dim=0).cpu()
        hidden_states_dict[k] = layer_states

    return hidden_states_dict, torch.stack(labels, dim=0)


if __name__ == "__main__":
    # Parse Args
    args = parse_arguments()

    # Load up model and image preprocessor
    if "scratch" not in args.model:
        processor = AutoImageProcessor.from_pretrained(args.model)

    if "mae" in args.model:
        model = ViTModel.from_pretrained(args.model).eval().to("cuda").to(dtype=torch.bfloat16)
    elif "scratch_" in args.model:
        patch_size = int(args.model.split("_")[1])
        config = ViTConfig.from_pretrained(f"google/vit-base-patch{patch_size}-224-in21k")
        processor = AutoImageProcessor.from_pretrained(f"google/vit-base-patch{patch_size}-224-in21k")
        model = ViTModel(config).eval().to("cuda").to(dtype=torch.bfloat16)
    else:
        model = AutoModel.from_pretrained(args.model).eval().to("cuda").to(dtype=torch.bfloat16)        

    if "clip" in args.model:
        model = model.vision_model

    output_dir = os.path.join(args.outfolder, args.model)
    os.makedirs(output_dir, exist_ok=True)
    
    args_dict = vars(args)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(args_dict, f)

    train_features, train_labels = process_dataset(args, "train", processor, model)
    val_features, val_labels = process_dataset(args, "val", processor, model)
    test_features, test_labels = process_dataset(args, "test", processor, model)

    torch.save(train_features, os.path.join(output_dir, "train_features.pth"))
    torch.save(val_features, os.path.join(output_dir, "val_features.pth"))
    torch.save(test_features, os.path.join(output_dir, "test_features.pth"))

    torch.save(train_labels, os.path.join(output_dir, "train_labels.pth"))
    torch.save(val_labels, os.path.join(output_dir, "val_labels.pth"))
    torch.save(test_labels, os.path.join(output_dir, "test_labels.pth"))