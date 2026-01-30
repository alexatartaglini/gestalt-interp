import argparse
import os
import json

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import StepLR

from transformers import AutoImageProcessor, AutoModel

from PIL import Image
from tqdm import tqdm

import utils.dataset
import utils.probe
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
        "-batch_size",
        default=256,
        type=int,
    )

    parser.add_argument(
        "-lr",
        default=.001
    )

    parser.add_argument(
        "-step_size",
        default=8
    )

    parser.add_argument(
        "-scheduler_gamma",
        default=.2
    )

    parser.add_argument(
        "-max_epochs",
        default=16
    )

    parser.add_argument("-layer", default=10, type=int, help="layer to train probe on")

    parser.add_argument(
        "-outfolder", default="./results", type=str, help="Directory to store results"
    )

    args = parser.parse_args()
    return args


def train_probe(args, probe_size, trainloader, valloader):
    
    results = {
        "epoch": [],
        "train loss": [],
        "train acc": [],
        "val loss": [],
        "val acc": []
    }

    probe = utils.probe.QuadraticProbe(probe_size).to("cuda").to(dtype=torch.bfloat16)
    optimizer = optim.Adam(probe.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.scheduler_gamma) 
    
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.max_epochs):
        epoch_train_loss = []
        epoch_train_correct = []
        for hidden_states, labels in tqdm(trainloader):
            optimizer.zero_grad()
            hidden_states = hidden_states.to("cuda").to(dtype=torch.bfloat16)

            labels = labels.to("cuda")
            pairwise_labels = (labels.unsqueeze(1) == labels.unsqueeze(2))

            # mask out any pair where either element is background
            valid_mask = (labels.unsqueeze(1) != 0) & (labels.unsqueeze(2) != 0)
            pairwise_labels = pairwise_labels[valid_mask].reshape(-1).float()

            # Compute pairwise probe results
            pairwise_matrices = []
            for batch_item in hidden_states:
                pairwise_matrices.append(probe.forward(batch_item, batch_item))

            outputs = torch.stack(pairwise_matrices, dim=0)[valid_mask].reshape(-1)
            predictions = torch.sigmoid(outputs) > 0.5


            loss = criterion(outputs, pairwise_labels)
            loss.backward()

            optimizer.step()

            epoch_train_correct.append((predictions == pairwise_labels.bool()).detach().cpu().numpy())
            epoch_train_loss.append(loss.detach().cpu().numpy())

        scheduler.step()

        epoch_val_loss = []
        epoch_val_correct = []
        with torch.no_grad():
            for hidden_states, labels in tqdm(valloader):
                hidden_states = hidden_states.to("cuda").to(dtype=torch.bfloat16)

                labels = labels.to("cuda")
                pairwise_labels = (labels.unsqueeze(1) == labels.unsqueeze(2))

                # mask out any pair where either element is background
                valid_mask = (labels.unsqueeze(1) != 0) & (labels.unsqueeze(2) != 0)
                pairwise_labels = pairwise_labels[valid_mask].reshape(-1).float()

                # Compute pairwise probe results
                pairwise_matrices = []
                for batch_item in hidden_states:
                    pairwise_matrices.append(probe.forward(batch_item, batch_item))

                outputs = torch.stack(pairwise_matrices, dim=0)[valid_mask].reshape(-1)
                predictions = torch.sigmoid(outputs) > 0.5

                loss = criterion(outputs, pairwise_labels)
                epoch_val_correct.append((predictions == pairwise_labels.bool()).detach().cpu().numpy())
                epoch_val_loss.append(loss.detach().cpu().numpy())
        
        results["epoch"].append(epoch)
        results["train loss"].append(float(np.mean(epoch_train_loss)))
        results["train acc"].append(float(np.concatenate(epoch_train_correct).mean()))
        results["val loss"].append(float(np.mean(epoch_val_loss)))
        results["val acc"].append(float(np.concatenate(epoch_val_correct).mean()))
        print(f"Epoch {epoch+1} train loss: {np.mean(epoch_train_loss):.4f}, train acc: {np.concatenate(epoch_train_correct).mean():.4f}, val loss: {np.mean(epoch_val_loss):.4f}, val acc: {np.concatenate(epoch_val_correct).mean():.4f}")
    return probe, results


def eval_probe(probe, testloader):
    
    results = {
        "loss": [],
        "acc": [],
    }

    probe = probe.eval()
    with torch.no_grad():
        criterion = nn.BCEWithLogitsLoss()

        mean_loss = []
        correct = []
        for hidden_states, labels in tqdm(testloader):
            hidden_states = hidden_states.to("cuda").to(dtype=torch.bfloat16)
            labels = labels.to("cuda")
            pairwise_labels = (labels.unsqueeze(1) == labels.unsqueeze(2))

            # mask out any pair where either element is background
            valid_mask = (labels.unsqueeze(1) != 0) & (labels.unsqueeze(2) != 0)
            pairwise_labels = pairwise_labels[valid_mask].reshape(-1).float()

            # Compute pairwise probe results
            pairwise_matrices = []
            for batch_item in hidden_states:
                pairwise_matrices.append(probe.forward(batch_item, batch_item))

            outputs = torch.stack(pairwise_matrices, dim=0)[valid_mask].reshape(-1)
            predictions = torch.sigmoid(outputs) > 0.5

            loss = criterion(outputs, pairwise_labels)
            correct.append((predictions == pairwise_labels.bool()).detach().cpu().numpy())
            mean_loss.append(loss.detach().cpu().numpy())
        
        results["loss"].append(float(np.mean(mean_loss)))
        results["acc"].append(float(np.concatenate(correct).mean()))

    return results

if __name__ == "__main__":
    # Parse Args
    args = parse_arguments()


    trainset = utils.dataset.ProbeDataset(
        path=f"./cache/{args.model}/{args.layer}/train"
    )
    valset = utils.dataset.ProbeDataset(
        path=f"./cache/{args.model}/{args.layer}/val"
    )
    testset = utils.dataset.ProbeDataset(
        path=f"./cache/{args.model}/{args.layer}/test"
    )

    probe_size = trainset[0][0].shape[-1]

    trainloader = DataLoader(trainset, batch_size = args.batch_size, shuffle = True)
    valloader = DataLoader(valset, batch_size = args.batch_size, shuffle = False)
    testloader = DataLoader(testset, batch_size = args.batch_size, shuffle = False)

    probe, results = train_probe(args, probe_size, trainloader, valloader)
    test_results = eval_probe(probe, testloader)

    output_dir = os.path.join(args.outfolder, args.model, str(args.layer))
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f)
    with open(os.path.join(output_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f)

    args_dict = vars(args)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(args_dict, f)

    torch.save(probe.state_dict(), os.path.join(output_dir, "probe.pth"))