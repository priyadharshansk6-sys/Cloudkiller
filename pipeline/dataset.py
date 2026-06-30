
import os
from glob import glob

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from config import *


class CloudRemovalDataset(Dataset):

    def __init__(self, transform=None):

        self.transform = transform

        print(f"\n[Dataset Debug] DATASET_ROOT: {DATASET_ROOT}")
        print(f"[Dataset Debug] TRAIN_RED_DIR: {TRAIN_RED_DIR}")
        print(f"[Dataset Debug] TRAIN_GT_DIR: {TRAIN_GT_DIR}")

        self.red_files = sorted(
            glob(os.path.join(TRAIN_RED_DIR, "*.TIF"))
        )
        print(f"[Dataset Debug] Found {len(self.red_files)} RED files.")

        self.green_files = sorted(
            glob(os.path.join(TRAIN_GREEN_DIR, "*.TIF"))
        )
        print(f"[Dataset Debug] Found {len(self.green_files)} GREEN files.")

        self.blue_files = sorted(
            glob(os.path.join(TRAIN_BLUE_DIR, "*.TIF"))
        )
        print(f"[Dataset Debug] Found {len(self.blue_files)} BLUE files.")

        self.nir_files = sorted(
            glob(os.path.join(TRAIN_NIR_DIR, "*.TIF"))
        )
        print(f"[Dataset Debug] Found {len(self.nir_files)} NIR files.")

        self.mask_files = sorted(
            glob(os.path.join(TRAIN_GT_DIR, "*.TIF"))
        )
        print(f"[Dataset Debug] Found {len(self.mask_files)} GT files.")

        self.length = len(self.red_files)
        print(f"[Dataset Debug] Dataset length (based on RED files): {self.length}")

        # Add a check to prevent empty dataset issues
        if self.length == 0:
            raise RuntimeError("No image files found. Check DATASET_ROOT and TRAIN_X_DIR paths in config.py")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        red = np.array(Image.open(self.red_files[idx]), dtype=np.float32) / 65535.0
        green = np.array(Image.open(self.green_files[idx]), dtype=np.float32) / 65535.0
        blue = np.array(Image.open(self.blue_files[idx]), dtype=np.float32) / 65535.0
        nir = np.array(Image.open(self.nir_files[idx]), dtype=np.float32) / 65535.0

        image = np.stack(
            [red, green, blue, nir],
            axis=-1
        )

        mask = np.array(
            Image.open(self.mask_files[idx]),
            dtype=np.float32
        )

        mask = (mask > 0).astype(np.float32)

        if self.transform:

            transformed = self.transform(
                image=image,
                mask=mask
            )

            image = transformed["image"]
            mask = transformed["mask"]

            mask = mask.unsqueeze(0)

        else:

            image = torch.tensor(
                image.transpose(2, 0, 1),
                dtype=torch.float32
            )

            mask = torch.tensor(
                mask,
                dtype=torch.float32
            ).unsqueeze(0)

        return image, mask
