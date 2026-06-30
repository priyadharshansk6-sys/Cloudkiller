
import torch
import torch.nn as nn
import torch.nn.functional as F
# The torchvision import for PerceptualLoss is not needed for DiceLoss, FocalLoss, and CombinedLoss.
# from torchvision import models 

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs) # Apply sigmoid for probabilities
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.bce_loss(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        return self.dice_weight * self.dice_loss(inputs, targets) + self.focal_weight * self.focal_loss(inputs, targets)

import os
import datetime

losses_file_path = 'losses.py'
print(f"\n--- Status of {losses_file_path} ---")
if os.path.exists(losses_file_path):
    print(f"Exists: True")
    print(f"Absolute Path: {os.path.abspath(losses_file_path)}")
    print(f"Size: {os.path.getsize(losses_file_path) / (1024 * 1024):.4f} MB")
    last_modified_timestamp = os.path.getmtime(losses_file_path)
    print(f"Last Modified: {datetime.datetime.fromtimestamp(last_modified_timestamp)}")
else:
    print(f"Exists: False")
