import pandas as pd
import numpy as np
import os
import yaml
import copy

def get_top_heads_before_layer(df, layer, n_heads=5):
    # Cols are layers, rows are head_idxs
    values = df.to_numpy()
    values = values[:, :layer+1] # Get rid of all heads above the probe layer

    flat_indices = np.argsort(values.flatten())[-5:][::-1]
    head_indices, layer_indices = np.unravel_index(flat_indices, values.shape)

    head_array = []
    for i in range(n_heads):
        head_array.append(str(layer_indices[i]) + "_" + str(head_indices[i]))
    return head_array


if __name__ == "__main__":

    N_HEADS = 5

    with open("base.yml", 'r') as stream:
        base = yaml.load(stream, Loader=yaml.FullLoader)

    for model in ["clip_16", "clip_32", "dino_16", "dinov2_14", "imagenet_16", "imagenet_32", "mae_16"]:
        df = pd.read_csv(f"../cache/continuity_scores/rotation_normalized_amplitude_{model}_curves.csv")

        patch_size = int(model.split("_")[1])
        tl_model = model.split("_")[0]

        pretrain_paths = {
            "mae": "facebook/vit-mae-base",  # patch size = 16
            "dino": f"facebook/dino-vitb{patch_size}",
            "imagenet": f"google/vit-base-patch{patch_size}-224-in21k",
            "clip": f"openai/clip-vit-base-patch{patch_size}",
            "dinov2": "facebook/dinov2-base",
        }

        probe_path = pretrain_paths[tl_model]

        for layer in range(12):
            heads = get_top_heads_before_layer(df, layer, N_HEADS)

            config = copy.deepcopy(base)
            config_name = model + "_" + str(layer)

            config["tl_model"] = tl_model
            config["patch_size"] = patch_size
            config["probe_path"] = probe_path
            config["layer"] = layer
            config["ablation_heads"] = heads

            os.makedirs(os.path.join("curves", model), exist_ok=True)
            config_path = os.path.join("curves", model, config_name + ".yaml")
            with open(config_path, 'w') as cfg_file:
                yaml.dump(config, cfg_file)
