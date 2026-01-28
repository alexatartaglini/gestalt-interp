import argparse
import os
import json

import numpy as np

import torch
from torch.utils.data import DataLoader, Subset

from transformers import AutoImageProcessor, AutoModel

from tqdm import tqdm

import utils.dataset
import utils.processing


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        default="facebook/dinov2-base",
        help="Model to use as backbone for analysis",
    )

    parser.add_argument(
        "-num_training_images",
        default=20000,
        type=int,
        help="Number of training images to use",
    )

    parser.add_argument(
        "-num_val_images",
        default=128,
        type=int,
        help="Number of validation images to use",
    )

    parser.add_argument(
        "-batch_size",
        default=128,
        type=int,
    )

    parser.add_argument(
        "-num_sample_patches",
        default=64,
        help="Sample a random number N of patches per image to supervise probe"
    )

    parser.add_argument("-layer", default=10, type=int, help="layer to train probe on")

    parser.add_argument(
        "-outfolder", default="./cache", type=str, help="Directory to store results"
    )

    parser.add_argument(
        "-random_seed",
        default=19
    )

    args = parser.parse_args()
    return args


def featurize_data(args, processor, model, dataloader):
    
    labels = []
    hidden_states = []
    model.eval()
    with torch.no_grad():  # Disable gradient computation to save memory
        for images, segmentation_masks in tqdm(dataloader):
            batch_hidden_states = utils.processing.get_hidden_states(model, processor, images, args.layer)
            batch_labels = utils.processing.extract_patch_labels(model.config.patch_size, np.stack(segmentation_masks, axis=0))

            sampled_indices = np.random.choice(batch_labels.shape[1], size=args.num_sample_patches, replace=False)
            
            batch_hidden_states = batch_hidden_states[:, sampled_indices].detach().to("cpu").to(dtype=torch.bfloat16)
            batch_labels = batch_labels[:, sampled_indices].detach().to("cpu").to(dtype=torch.bfloat16)

            labels.append(batch_labels)
            hidden_states.append(batch_hidden_states)
            
            # Clear GPU cache periodically to prevent fragmentation
            if len(labels) % 50 == 0:
                torch.cuda.empty_cache()

    return torch.concat(hidden_states, dim=0), torch.concat(labels, dim=0)


if __name__ == "__main__":
    # Parse Args
    args = parse_arguments()

    np.random.seed(args.random_seed)
    # Load up model and image preprocessor
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval().to("cuda").to(dtype=torch.bfloat16)

    dataset = utils.dataset.ADE20KSegmentation(
        cache="./cache", image_set="train", max_examples=None, processor=processor
    )
    testset = utils.dataset.ADE20KSegmentation(
        cache="./cache", image_set="validation", max_examples=None, processor=processor
    )

    set_indices = np.random.choice(range(len(dataset)), size=args.num_training_images + args.num_val_images, replace=False)
    trainset = Subset(dataset, set_indices[:args.num_training_images])
    valset = Subset(dataset, set_indices[args.num_training_images:])

    trainloader = DataLoader(trainset, batch_size = args.batch_size, shuffle = False, collate_fn=utils.dataset.collate_fn, num_workers=8, pin_memory=True)
    valloader = DataLoader(valset, batch_size = args.batch_size, shuffle = False, collate_fn=utils.dataset.collate_fn, num_workers=8, pin_memory=True)
    testloader = DataLoader(testset, batch_size = args.batch_size, shuffle = False, collate_fn=utils.dataset.collate_fn, num_workers=8, pin_memory=True)

    train_features, train_labels = featurize_data(args, processor, model, trainloader)
    print(train_features.shape, train_labels.shape)
    val_features, val_labels = featurize_data(args, processor, model, valloader)
    test_features, test_labels = featurize_data(args, processor, model, testloader)

    output_dir = os.path.join(args.outfolder, args.model, str(args.layer))
    os.makedirs(output_dir, exist_ok=True)
    
    args_dict = vars(args)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(args_dict, f)

    torch.save(train_features, os.path.join(output_dir, "train_features.pth"))
    torch.save(val_features, os.path.join(output_dir, "val_features.pth"))
    torch.save(test_features, os.path.join(output_dir, "test_features.pth"))

    torch.save(train_labels, os.path.join(output_dir, "train_labels.pth"))
    torch.save(val_labels, os.path.join(output_dir, "val_labels.pth"))
    torch.save(test_labels, os.path.join(output_dir, "test_labels.pth"))