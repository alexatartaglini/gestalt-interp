import torch
from torch.utils.data import Dataset
import numpy as np
from . import processing


from datasets import load_dataset

class ADE20KSegmentation(Dataset):
    def __init__(self, cache="../cache/", image_set="validation", max_examples=None, processor=None):
        self.dataset = load_dataset("1aurent/ADE20K", cache_dir=cache)[image_set]
        self.processor=processor

        print(len(self.dataset))
        if max_examples:
            # Use select instead of slicing to maintain proper dataset structure
            indices = list(range(min(max_examples, len(self.dataset))))
            self.dataset = self.dataset.select(indices)
    
    def __getitem__(self, idx):
        
        image = np.array(self.dataset[idx]["image"])
        segmentations = np.array(self.dataset[idx]["segmentations"][0])

        # Turn segmentations into a (H, W) matrix with value determining object instance
        instance_channel = segmentations[:, :, 2]
        instance_mask = np.zeros_like(instance_channel)

        non_bg = instance_channel > 0  # Remaps non Background pixels to instance labels
        instance_mask[non_bg] = np.unique(instance_channel[non_bg], return_inverse=True)[1] + 1
        instance_mask = processing.preprocess_segmentation(instance_mask, self.processor)
    
        return {
            "image": image, 
            "segmentation": instance_mask
        }

    def __len__(self):
        return len(self.dataset)
    
class ProbeDataset(Dataset):
    def __init__(self, path):
        self.x = torch.load(path + "_features.pth")
        self.y = torch.load(path + "_labels.pth")

    def __getitem__(self, idx):
        
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)
    
    
def collate_fn(batch):
    # Need this because of variable sized images and masks
    images = [b["image"] for b in batch]
    masks = [b["segmentation"] for b in batch]
    return images, masks