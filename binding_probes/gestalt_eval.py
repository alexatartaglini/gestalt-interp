import argparse
import os
import json
import yaml

from PIL import Image
import numpy as np

import torch
from transformers import AutoImageProcessor, AutoModel

from tqdm import tqdm
from utils.probe import QuadraticProbe

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        default="facebook/dinov2-base",
        help="Model to use as backbone for analysis",
    )


    parser.add_argument(
        "-dataset",
        default="./cache/scrambled_objects",
        type=str,
        help="Number of validation images to use",
    )

    parser.add_argument("-layer", default=10, type=int, help="layer to train probe on")

    parser.add_argument(
        "-outfolder", default="./results", type=str, help="Directory to store results"
    )

    args = parser.parse_args()
    return args


def cfg2labels(file):
    bg_patches = file["bg_patch_indices"] # 1 indexed to account for cls token
    obj_patches = file["object_patch_indices"]
    bg_patches = [idx - 1 for idx in bg_patches]
    obj_patches = [idx - 1 for idx in obj_patches]

    total_patches = np.max(bg_patches + obj_patches) + 1
    obj_mask = np.zeros(total_patches)
    obj_mask[obj_patches] = 1 # mask segmenting object from background
    return torch.Tensor(obj_mask)


def img2states(img, processor, model, layer):
    inputs = processor(img, return_tensors="pt")
    inputs = {k: v.cuda().to(dtype=torch.bfloat16) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad():  # Ensure no gradients are computed
        hidden_states = model(**inputs, output_hidden_states=True).hidden_states[layer+1]
    return hidden_states[0, 1:, :]  # Always only one image here

def process_dataset(args, split, processor, model, probe):

    patch_size = model.config.patch_size
    model.eval()

    data = os.listdir(args.dataset)

    probe_acc = []
    chance_diff_object = []

    for datum in tqdm(data):
        if datum == "cfg.yaml":
            continue

        if split == "base":
            img_path = os.path.join(args.dataset, datum, "base.png")
        else:
            img_path = os.path.join(args.dataset, datum, f"{split}_PATCHSIZE-{patch_size}.png")
        cfg_path = os.path.join(args.dataset, datum, f"cfg_PATCHSIZE-{patch_size}.yaml")

        labels = cfg2labels(yaml.safe_load(open(cfg_path, "r")))
        hidden_states = img2states(Image.open(img_path).convert("RGB"), processor, model, args.layer)

        logits = probe.forward(hidden_states, hidden_states).reshape(-1)
        predictions = torch.sigmoid(logits) > 0.5
        pairwise_labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).reshape(-1).float()


        probe_acc.append(torch.mean((predictions.cpu() == pairwise_labels).float().detach()).item())
        chance_diff_object.append(torch.mean((pairwise_labels == 0).float()).item())
    return np.mean(probe_acc), np.mean(chance_diff_object)


if __name__ == "__main__":
    # Parse Args
    args = parse_arguments()

    # Load up model and image preprocessor
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval().to("cuda").to(dtype=torch.bfloat16)

    probe = QuadraticProbe(model.config.hidden_size)
    probe_state_dict = torch.load(f"./results/{args.model}/{args.layer}/probe.pth")
    probe.load_state_dict(probe_state_dict)
    probe.eval().to("cuda").to(dtype=torch.bfloat16)


    results = {}
    for dataset in ["base", "location", "orientation"]:
        accuracy, chance_diff_object = process_dataset(args, dataset, processor, model, probe)
        results[dataset] = accuracy
        results["chance_diff_object"] = chance_diff_object
        print(results)

    output_dir = os.path.join(args.outfolder, args.model, str(args.layer))
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "gestalt_eval.json"), "w") as f:
        json.dump(results, f)