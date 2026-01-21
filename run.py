import argparse
import os
import tqdm

import torch
import pandas as pd

from utils.model_utils import Model
from utils.dataset_utils import ContinuityDataset


def main(args):
    # Prepare paths
    result_path = f"{args.model_pretrain}_{args.model_patch_size}"
    if args.model_pretrain == "scratch":
        result_path += f"_{args.model_architecture_type}"
    result_path += f"/{args.dataset_path}"

    full_dataset_path = os.path.join(args.dataset_root, args.dataset_dir, args.dataset_path)
    full_result_path = os.path.join(args.result_root, args.result_dir, result_path)
    os.makedirs(full_result_path, exist_ok=True)

    # Load dataset
    dataset = ContinuityDataset.load_from_path(full_dataset_path)
    dataset.filter_by_patch_size(args.model_patch_size)
    if args.control:
        dataset.filter_by_control(args.control)

    # Load model
    model = Model(
        pretrain=args.model_pretrain,
        patch_size=args.model_patch_size,
        scratch_architecture=args.model_architecture_type,
    )
    model.eval()
    n_heads = model.model.cfg.n_heads
    n_layers = model.model.cfg.n_layers

    # Prepare result template
    generic_result_template = {
        "model_pretrain": args.model_pretrain,
        "model_patch_size": args.model_patch_size,
        "model_architecture_type": args.model_architecture_type,
        "full_dataset_path": full_dataset_path,
        "full_result_path": full_result_path,
        "dataset_path": args.dataset_path,
    }

    # Run all experiments
    for experiment in tqdm.tqdm(dataset, desc="Running all experiments"):
        # Set up result dictionary
        cfg = experiment.cfg
        experiment_result = {
            **generic_result_template,
            **cfg,
        }
        result_name = cfg["experiment_id"] + "_" + cfg["experiment_path"].split("/")[-1] + ".csv"
        experiment_result["full_result_path"] = os.path.join(full_result_path, result_name)
        experiment_result.pop("neighbor_tokens")
        experiment_result.pop("perimeter_tokens")
        experiment_result.pop("target_tokens")

        # Get token info
        max_aligned_index = experiment.get_max_aligned_index()
        all_src_token_info = experiment.src_token_info
        all_dest_token_info_per_src = experiment.dest_token_info_per_src

        # Run experiment
        all_results = []
        src_dest_layer_head_attn = {}
        src_to_dest_results = {}

        for src, src_info in all_src_token_info.items():
            for dest_info in all_dest_token_info_per_src[src]:
                dest = dest_info["dest_token"]
                
                src_to_dest_results[(src, dest)] = {
                    **experiment_result,
                    **src_info,
                    **dest_info,
                    "layer": None,
                    "head": None,
                    "attns": [],
                    "max_aligned_index": max_aligned_index,
                    "steps": experiment.steps,
                }

                for layer in range(n_layers):
                    for head in range(n_heads):
                        src_dest_layer_head_attn[(src, dest, layer, head)] = []

        model.eval()
        with torch.no_grad():
            for stimulus in tqdm.tqdm(experiment, desc="Running experiment"):
                _, cache = model.run_with_cache(stimulus, remove_batch_dim=True)

                for layer in range(n_layers):
                    for head in range(n_heads):
                        layer_head_attn = cache["pattern", layer, "attn"]

                        for src, dest in src_to_dest_results.keys():
                            attn = layer_head_attn[head, src, :].squeeze(0) * 100
                            attn = attn[dest].cpu().item()
                            src_dest_layer_head_attn[(src, dest, layer, head)].append(attn)

            for src, dest, layer, head in src_dest_layer_head_attn.keys():
                result = src_to_dest_results[(src, dest)].copy()
                result["layer"] = layer
                result["head"] = head
                result["attns"] = src_dest_layer_head_attn[(src, dest, layer, head)]
                all_results.append(result)

        all_results = pd.DataFrame(all_results)
        all_results.to_csv(experiment_result["full_result_path"], index=False)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="/home/alexart/gestalt-interp")
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--dataset_path", type=str, default="filled_shapes")
    parser.add_argument("--result_root", type=str, default="/data/alexart/gestalt-interp")
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--model_pretrain", type=str, default="clip")
    parser.add_argument("--model_patch_size", type=int, default=32)
    parser.add_argument("--model_architecture_type", type=str, default="imagenet")
    parser.add_argument("--control", type=str, default=None)

    args = parser.parse_args()
    main(args)