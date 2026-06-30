
import os
import random
from glob import glob

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from config import *
from dataset import CloudRemovalDataset # Base dataset for loading individual bands

class ReconstructionDatasetV2(Dataset):
    """
    Paired Dataset for GenAI Restoration.
    Target: Clear Tiles (0% Cloud)
    Input: Clear Tile + Real Cloud Mask (from a different cloudy tile)
    """
    def __init__(self, transform=None):
        self.base_dataset = CloudRemovalDataset(transform=None) # Raw access to original dataset
        self.transform = transform

        print("[V2 Audit] Scanning for clear and cloudy templates...")
        self.clear_indices = [] # Indices of 100% clear tiles
        self.cloudy_indices = [] # Indices of tiles with real cloud masks

        # Iterate through the base dataset to identify clear and cloudy tiles
        for i in range(len(self.base_dataset)):
            mask_path = self.base_dataset.mask_files[i]
            mask = np.array(Image.open(mask_path))
            if np.max(mask) == 0: # Check if the mask contains any cloud pixels (value > 0)
                self.clear_indices.append(i)
            else:
                self.cloudy_indices.append(i)

        if not self.clear_indices:
            raise RuntimeError("No clear tiles found in the dataset for reconstruction targets.")
        if not self.cloudy_indices:
            raise RuntimeError("No cloudy tiles found for generating real cloud masks.")

        print(f"[V2 Audit] Found {len(self.clear_indices)} clear targets and {len(self.cloudy_indices)} real cloud masks.")

    def __len__(self):
        # The length of the dataset is determined by the number of available clear tiles,
        # as each clear tile will serve as a ground truth target.
        return len(self.clear_indices)

    def __getitem__(self, idx):
        # 1. Get a Clear Target Image (which will be our ground truth)
        target_original_idx = self.clear_indices[idx]
        target_img_bands, _ = self.base_dataset[target_original_idx] # (4, H, W) RGB + NIR, normalized

        # 2. Get a random REAL Cloud Mask template from a cloudy tile
        random_cloudy_original_idx = random.choice(self.cloudy_indices)
        _, real_mask = self.base_dataset[random_cloudy_original_idx] # (1, H, W) binary mask

        # 3. Create Cloudy Input Image (RGB + NIR) by applying the real_mask to the clear target image
        # This simulates a cloudy image where the clear target is obscured by a real cloud pattern.
        # The output `cloudy_img` should be 4 channels (RGB + NIR).
        cloudy_img = target_img_bands * (1 - real_mask) # Zero out pixels under the cloud mask

        # 4. Concatenate the cloudy image (4 channels) with the real_mask (1 channel) to form a 5-channel input
        # This 5-channel tensor (R, G, B, NIR, Mask) will be the input to the Generator.
        input_tensor = torch.cat([cloudy_img, real_mask], dim=0) # Result: (5, H, W)

        # Apply any global transforms if provided (e.g., augmentation)
        # Note: Current implementation of CloudRemovalDataset already applies initial transforms.
        # This 'transform' here would be for further augmentation specific to reconstruction if needed.
        if self.transform:
            # Need to convert back to PIL Image if transform expects it, then back to tensor
            # For simplicity, assuming `transform` operates on tensors or is already handled by base_dataset
            # For now, we'll keep the transform logic simple or assume it's mostly handled upstream.
            pass

        return input_tensor, target_img_bands


import os
import datetime

recon_dataset_file_path = 'reconstruction_dataset_v2.py'
print(f"\n--- Status of {recon_dataset_file_path} ---")
if os.path.exists(recon_dataset_file_path):
    print(f"Exists: True")
    print(f"Absolute Path: {os.path.abspath(recon_dataset_file_path)}")
    print(f"Size: {os.path.getsize(recon_dataset_file_path) / (1024 * 1024):.4f} MB")
    last_modified_timestamp = os.path.getmtime(recon_dataset_file_path)
    print(f"Last Modified: {datetime.datetime.fromtimestamp(last_modified_timestamp)}")
else:
    print(f"Exists: False")
