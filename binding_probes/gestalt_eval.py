"""
This script runs a pretrained probe over the test set of blob binding stimuli and
assesses the importance of particular attention heads on this representations via
ablation experiments.

Specifically, this file will:
- Assess performance of a pretrained probe on the standard blob binding test set
- Assess performance of a pretrained probe on two control sets, which isolate the effect
        of continuity on object binding (rather than patch content, per se)
- Ablate a set of heads that we suspect are implicated in probe behavior, assess performance
        on the base dataset
- Ablate N disjoint set of heads, assess performance on the base dataset
"""

import os
import json
import yaml
import argparse
import sys
from collections import defaultdict

from PIL import Image
import numpy as np

import torch

from tqdm import tqdm
from utils.probe import QuadraticProbe
from utils.model_utils import Model
from utils import processing


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--tl_model",
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
        "--dataset",
        default="./cache/scrambled_blobs/test",
        type=str,
        help="The directory containing test objects",
    )

    parser.add_argument("--probe_path", default="facebook/dinov2-base")

    parser.add_argument("--layer", type=int)

    parser.add_argument(
        "--outfolder", default="./results", type=str, help="Directory to store results"
    )

    parser.add_argument(
        "--ablation_heads",
        nargs="+",
        default=["6_8", "8_8"],
        type=str,
        help="A list of Layer_Head-Index strings to ablate (all at once!)",
    )

    parser.add_argument(
        "--n_controls",
        default=100,
        help="Number of random disjoint head sets to ablate",
    )
    args = parser.parse_args()
    return args


def get_config():
    # Load config file from command line arg
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="where to load YAML configuration",
        metavar="FILE",
    )

    argv = sys.argv[1:]

    args, _ = parser.parse_known_args(argv)

    if not hasattr(args, "config"):
        raise ValueError("Must include path to config file")
    else:
        with open(args.config, "r") as stream:
            return yaml.load(stream, Loader=yaml.FullLoader)


def retrieve_states_with_intervention(img, model, layer, head_strs=None, mean_acts=None):
    if head_strs is None:
        _, cache = model.run_with_cache(img)
    else:
        _, cache = model.run_with_head_ablation(img, head_strs, mean_acts)
    return cache[f"blocks.{layer}.hook_resid_post"][0, 1:, :]


def parse_head_list(head_list):
    # Head List is a list of strings, indicating layer_head-idx
    # Parse this for sampling disjoint head sets
    layer2heads = defaultdict(list)
    for head in head_list:
        head = head.split("_")
        layer = int(head[0])
        head_idx = int(head[1])
        layer2heads[layer].append(head_idx)
    return layer2heads


def sample_disjoint_head_set(layer2heads, n_heads=12):
    # Ablated heads are stored in a dictionary mapping
    # layer -> list(heads)
    # Sample a disjoint set of heads for control ablation,
    # where the disjoint set contains the same number of heads per layer
    sampled_head_list = []
    for layer in layer2heads.keys():
        selected_heads = set(layer2heads[layer])
        n_to_sample = len(selected_heads)
        disjoint_heads = list(set(list(range(n_heads))) - selected_heads)
        if len(disjoint_heads) < n_to_sample:
            raise ValueError(
                f"Number of heads to sample is more than the number of disjoint heads at layer {layer}"
            )
        sampled_disjoint_heads = np.random.choice(disjoint_heads, size=n_to_sample, replace=False)
        for head in sampled_disjoint_heads:
            sampled_head_list.append(str(layer) + "_" + str(head))
    return sampled_head_list


def collect_activations(args, model):
    """Run model over the base test set, and collect mean activations"""
    model.eval()
    data = os.listdir(args["mean_dataset"])

    count = 0
    dataset_mean_acts = None
    for datum in tqdm(data):
        if datum == "cfg.yaml":
            # Skip cfg file in directory
            continue

        img_path = os.path.join(args["mean_dataset"], datum, "base.png")
        acts = model.get_mean_head_activations(Image.open(img_path).convert("RGB"))

        if not dataset_mean_acts:
            # Initialize
            dataset_mean_acts = {layer: act.clone() for layer, act in acts.items()}
        else:
            for layer in acts:
                dataset_mean_acts[layer] += acts[layer]
        
        count += 1

    # Compute average
    for layer in dataset_mean_acts:
        dataset_mean_acts[layer] /= count

    return dataset_mean_acts


def evalutate_dataset(args, dataset, model, probe, heads=None, mean_acts=None):
    """Evaluate a particular dataset, optionally by ablating heads"""

    patch_size = model.model.cfg.patch_size
    model.eval()

    data = os.listdir(args["dataset"])

    probe_acc = []
    chance_diff_object = []

    for datum in tqdm(data):
        if datum == "cfg.yaml":
            # Skip cfg file in directory
            continue

        if dataset == "base":
            img_path = os.path.join(args["dataset"], datum, "base.png")
        else:
            img_path = os.path.join(
                args["dataset"], datum, f"{dataset}_PATCHSIZE-{patch_size}.png"
            )

        cfg_path = os.path.join(args["dataset"], datum, f"cfg_PATCHSIZE-{patch_size}.yaml")

        hidden_states = retrieve_states_with_intervention(
            Image.open(img_path).convert("RGB"),
            model,
            args["layer"],
            head_strs=heads,
            mean_acts=mean_acts
        )

        labels = processing.cfg2labels(yaml.safe_load(open(cfg_path, "r")))
        pairwise_labels = labels.unsqueeze(0) == labels.unsqueeze(1)

        # mask out any pair where either element is background
        valid_mask = (labels.unsqueeze(0) != 0) & (labels.unsqueeze(1) != 0)
        pairwise_labels = pairwise_labels[valid_mask].reshape(-1).float()

        logits = probe.forward(hidden_states, hidden_states)[valid_mask].reshape(-1)
        predictions = torch.sigmoid(logits) > 0.5

        probe_acc.append(
            torch.mean(
                (predictions.cpu() == pairwise_labels.bool()).float().detach()
            ).item()
        )
        chance_diff_object.append(torch.mean((pairwise_labels == 0).float()).item())
    return np.mean(probe_acc), np.mean(chance_diff_object)


if __name__ == "__main__":
    # Parse Args
    args = get_config()

    # Load up model and image preprocessor
    model = Model(
        pretrain=args["tl_model"],  # imagenet, dinov2, scratch
        patch_size=args["patch_size"],  # 14 for dinov2; 16 or 32 for clip/imagenet
    )
    model.to("cuda")

    probe = QuadraticProbe(model.model.cfg.d_model)
    probe_state_dict = torch.load(f"./results/{args["probe_path"]}/{args["layer"]}/probe.pth")
    probe.load_state_dict(probe_state_dict)
    probe.eval().to("cuda").to(dtype=torch.float32)

    dataset_list = ["base", "location", "orientation"]
    results = {}

    # First, assess probe performance on all three datasets without ablation
    print("Running standard probe on all datasets...")
    results["standard"] = {}
    for dataset in dataset_list:
        results["standard"][dataset] = {}
        accuracy, chance_diff_object = evalutate_dataset(
            args, dataset, model, probe, heads=None
        )
        results["standard"][dataset]["accuracy"] = accuracy
        results["standard"][dataset]["chance (different)"] = chance_diff_object

    # Next, Compute the mean activations for each attention head over the test set,
    # for performing mean ablations later
    print("Collecting mean head activations...")
    layer2activations = collect_activations(
        args, model
    )

    # Now, run mean ablations
    print("Running ablation on all datasets...")
    results["ablation"] = {}
    for dataset in ["base"]:
        results["ablation"][dataset] = {}
        accuracy, chance_diff_object = evalutate_dataset(
            args, dataset, model, probe, heads=args["ablation_heads"], mean_acts=layer2activations
        )
        results["ablation"][dataset]["accuracy"] = accuracy
        results["ablation"][dataset]["chance (different)"] = chance_diff_object

    print(f"Running {args["n_controls"]} Control Ablations")
    head_dict = parse_head_list(args["ablation_heads"])

    # Finally, run controls
    results["controls"] = {}
    for dataset in ["base"]:
        results["controls"][dataset] = defaultdict(list)

        print(f"Running controls on {dataset}...")

        for n in tqdm(range(args["n_controls"])):
            sampled_heads = sample_disjoint_head_set(head_dict)
            accuracy, chance_diff_object = evalutate_dataset(
                args, dataset, model, probe, heads=sampled_heads, mean_acts=layer2activations
            )
            results["controls"][dataset]["accuracy"].append(accuracy)
            results["controls"][dataset]["chance (different)"].append(chance_diff_object)

        output_dir = os.path.join(
            args["outfolder"], args["probe_path"], str(args["layer"]),
        )
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "gestalt_eval.json"), "w") as f:
            json.dump(results, f)

        print("Finished!")
