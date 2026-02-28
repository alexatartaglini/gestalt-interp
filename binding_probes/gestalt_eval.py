import argparse
import os
import json
import yaml

from PIL import Image
import numpy as np

import torch

from tqdm import tqdm
from utils.probe import QuadraticProbe
from utils.model_utils import Model

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        default="dinov2",
        help="Model to use as backbone for analysis",
    )

    parser.add_argument(
        "--patch_size",
        default=14,
        type=int,
        help="Patch size for model",
    )

    parser.add_argument(
        "-dataset",
        default="./cache/scrambled_objects/test",
        type=str,
        help="The directory containing test objects",
    )

    parser.add_argument("--probe_path", default="facebook/dinov2-base")

    parser.add_argument("-layer", default=10, type=int, help="layer to train probe on")

    parser.add_argument(
        "-outfolder", default="./results", type=str, help="Directory to store results"
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Enable head ablation"
    )

    parser.add_argument(
        "--ablate_all",
        action="store_true"
    )
    parser.add_argument(
        "--ablation_heads",
        nargs="+",
        default=["6_8", "8_8"],
        type=str,
        help="A list of Layer_Head-Index strings to ablate"
    )


    args = parser.parse_args()
    return args


def cfg2labels(file):
    obj1_patches = file["object1_patch_indices"]
    obj2_patches = file["object2_patch_indices"]

    bg_patches = file["bg_patch_indices"]

    total_patches = np.max(obj1_patches + obj2_patches + bg_patches)

    # Ensure no overlapping patches!
    filtered_obj1_patches = list(set(obj1_patches) - set(obj2_patches))
    filtered_obj2_patches = list(set(obj2_patches) - set(obj1_patches))

    # 0 index, getting rid of CLS token
    obj1_patches = [idx - 1 for idx in filtered_obj1_patches] 
    obj2_patches = [idx - 1 for idx in filtered_obj2_patches]

    obj_mask = np.zeros(total_patches)
    obj_mask[obj1_patches] = 1 # mask segmenting object 1 from background
    obj_mask[obj2_patches] = 2 # mask segmenting object 2 from background

    return torch.Tensor(obj_mask)


def img2states(img, model, layer, ablation, head_str=None):
    if not ablation:
        _, cache = model.run_with_cache(img)
    else:
        _, cache = model.run_with_head_ablation(img, head_str)
    return cache[f"blocks.{layer}.hook_resid_post"][0, 1:, :]


def process_dataset(args, split, model, probe, head=None):

    patch_size = model.model.cfg.patch_size
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

        hidden_states = img2states(Image.open(img_path).convert("RGB"), model, args.layer, args.ablation, head_str=head)

        labels = cfg2labels(yaml.safe_load(open(cfg_path, "r")))
        pairwise_labels = (labels.unsqueeze(0) == labels.unsqueeze(1))

        # mask out any pair where either element is background
        valid_mask = (labels.unsqueeze(0) != 0) & (labels.unsqueeze(1) != 0)
        pairwise_labels = pairwise_labels[valid_mask].reshape(-1).float()

        logits = probe.forward(hidden_states, hidden_states)[valid_mask].reshape(-1)
        predictions = torch.sigmoid(logits) > 0.5

        probe_acc.append(torch.mean((predictions.cpu() == pairwise_labels.bool()).float().detach()).item())
        chance_diff_object.append(torch.mean((pairwise_labels == 0).float()).item())
    return np.mean(probe_acc), np.mean(chance_diff_object)


if __name__ == "__main__":
    # Parse Args
    args = parse_arguments()

    # Load up model and image preprocessor
    model = Model(
                pretrain=args.model,  # imagenet, dinov2, scratch
                patch_size=args.patch_size,  # 14 for dinov2; 16 or 32 for clip/imagenet
            )
    model.to("cuda")
    
    probe = QuadraticProbe(model.model.cfg.d_model)
    probe_state_dict = torch.load(f"./results/{args.probe_path}/{args.layer}/probe.pth")
    probe.load_state_dict(probe_state_dict)
    probe.eval().to("cuda").to(dtype=torch.float32)


    if args.ablation:
        dataset_list = ["base"]

        if args.ablate_all:
            head_list = []
            for i in range(12):
                for j in range(12):
                    head_list.append(f"{i}_{j}")
        else:
            head_list = args.ablation_heads

    else:
        head_list = [None]
        dataset_list = ["base", "location", "orientation"]

    for head in head_list:
        results = {}
        for dataset in dataset_list:
            accuracy, chance_diff_object = process_dataset(args, dataset, model, probe, head=head)
            results[dataset] = accuracy
            results["chance_diff_object"] = chance_diff_object
            print(results)

        if args.ablation:
            output_dir = os.path.join(args.outfolder, args.probe_path, str(args.layer), "ablation", head)
        else:
            output_dir = os.path.join(args.outfolder, args.probe_path, str(args.layer), "standard")

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "gestalt_eval.json"), "w") as f:
            json.dump(results, f)