
import torch
import torch.nn.functional as F

def calculate_dice_coefficient(outputs, targets, smooth=1e-6):
    outputs = torch.sigmoid(outputs)
    outputs = outputs.view(-1)
    targets = targets.view(-1)

    intersection = (outputs * targets).sum()
    dice = (2. * intersection + smooth) / (outputs.sum() + targets.sum() + smooth)

    return dice

def calculate_iou(outputs, targets, smooth=1e-6):
    outputs = torch.sigmoid(outputs)
    outputs = outputs.view(-1)
    targets = targets.view(-1)

    intersection = (outputs * targets).sum()
    union = outputs.sum() + targets.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)

    return iou

def calculate_sam(img1, img2, eps=1e-8):
    """
    Calculate Spectral Angle Mapper (SAM) between two multi-band images.
    img1, img2: Tensors of shape (C, H, W)
    Returns: Mean spectral angle in radians.
    """
    # Reshape to (H*W, C) to treat each pixel as a spectral vector
    img1_flat = img1.permute(1, 2, 0).reshape(-1, img1.shape[0])
    img2_flat = img2.permute(1, 2, 0).reshape(-1, img2.shape[0])

    # Normalize vectors to unit length
    img1_norm = F.normalize(img1_flat, p=2, dim=1)
    img2_norm = F.normalize(img2_flat, p=2, dim=1)

    # Dot product
    dot_product = (img1_norm * img2_norm).sum(dim=1)

    # Clamp for numerical stability
    dot_product = torch.clamp(dot_product, -1.0 + eps, 1.0 - eps)

    # Calculate angle
    angles = torch.acos(dot_product)

    return torch.mean(angles)

import os
import datetime

metrics_file_path = 'metrics.py'
print(f"\n--- Status of {metrics_file_path} ---")
if os.path.exists(metrics_file_path):
    print(f"Exists: True")
    print(f"Absolute Path: {os.path.abspath(metrics_file_path)}")
    print(f"Size: {os.path.getsize(metrics_file_path) / (1024 * 1024):.4f} MB")
    last_modified_timestamp = os.path.getmtime(metrics_file_path)
    print(f"Last Modified: {datetime.datetime.fromtimestamp(last_modified_timestamp)}")
else:
    print(f"Exists: False")
