
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
import importlib
from tqdm import tqdm
import os

import config
import model
import losses
import metrics
import dataset # Import dataset to ensure it's in sys.modules

# Reload modules to ensure the latest changes are used
importlib.reload(config)
importlib.reload(model)
importlib.reload(losses)
importlib.reload(metrics)
importlib.reload(dataset)

from config import (
    BATCH_SIZE, NUM_INPUT_CHANNELS_SEGMENTATION, NUM_CLASSES_SEGMENTATION, LR, PATIENCE, DEVICE, transform
)
from model import UNet
from losses import CombinedLoss
from metrics import calculate_dice_coefficient, calculate_iou
from dataset import CloudRemovalDataset

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    loop = tqdm(dataloader, desc=f"Training", leave=False)

    for batch_idx, (images, masks) in enumerate(loop):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        with autocast(): # Enable mixed precision
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        dice, iou = calculate_batch_metrics(outputs, masks)
        running_dice += dice
        running_iou += iou

        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(dataloader)
    avg_dice = running_dice / len(dataloader)
    avg_iou = running_iou / len(dataloader)

    return avg_loss, avg_dice, avg_iou

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    loop = tqdm(dataloader, desc=f"Validation", leave=False)

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loop):
            images = images.to(device)
            masks = masks.to(device)

            with autocast(): # Enable mixed precision
                outputs = model(images)
                loss = criterion(outputs, masks)

            running_loss += loss.item()
            dice, iou = calculate_batch_metrics(outputs, masks)
            running_dice += dice
            running_iou += iou

            loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(dataloader)
    avg_dice = running_dice / len(dataloader)
    avg_iou = running_iou / len(dataloader)

    return avg_loss, avg_dice, avg_iou

# Function to calculate metrics from batches
def calculate_batch_metrics(outputs, targets):
    dice = calculate_dice_coefficient(outputs, targets)
    iou = calculate_iou(outputs, targets)
    return dice.item(), iou.item()

def main(additional_epochs=config.EPOCHS): # Use EPOCHS from config
    print(f"Using device: {DEVICE}")

    # Setup checkpoint directory
    CHECKPOINT_DIR = "/content/drive/MyDrive/CloudKillers_Checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Checkpoints will be saved to: {CHECKPOINT_DIR}")

    # 1. Dataset and DataLoaders
    full_dataset = CloudRemovalDataset(transform=config.transform) # Use transform from config

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42) # For reproducibility
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 2. Model, Loss, Optimizer, Scaler
    model = UNet(
        in_channels=NUM_INPUT_CHANNELS_SEGMENTATION,
        out_channels=NUM_CLASSES_SEGMENTATION
    ).to(DEVICE)

    criterion = CombinedLoss().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler() # For mixed precision

    # Determine paths
    best_model_path_gdrive = os.path.join(CHECKPOINT_DIR, 'best_unet.pth')

    # Resume logic
    current_best_val_dice = -1.0
    epochs_no_improve = 0
    start_training_epoch = 0 # Logical epoch counter for display

    if os.path.exists(best_model_path_gdrive):
        print(f"Found existing checkpoint at '{best_model_path_gdrive}'. Loading model weights and performing initial validation.")
        model.load_state_dict(torch.load(best_model_path_gdrive, map_location=DEVICE))

        # Perform initial validation to establish `current_best_val_dice` for early stopping
        _, initial_val_dice, _ = validate_one_epoch(model, val_loader, criterion, DEVICE)
        current_best_val_dice = initial_val_dice
        print(f"Initial validation Dice for loaded model: {current_best_val_dice:.4f}")
        print(f"Continuing training for {additional_epochs} additional epochs.")
    else:
        print(f"No existing checkpoint found at '{best_model_path_gdrive}'. Starting training from scratch for {additional_epochs} epochs.")


    # Scheduler (re-initialize for the new set of `additional_epochs`)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=additional_epochs # T_max for the current 'additional_epochs' run
    )

    # 3. Training Loop with Early Stopping and Model Saving
    print("\nStarting training loop...")
    for epoch_offset in range(additional_epochs):
        current_epoch_display = start_training_epoch + epoch_offset + 1
        print(f"\nEpoch {current_epoch_display}/{additional_epochs} (Additional {epoch_offset + 1}/{additional_epochs})") # Adjusted total epochs for display

        train_loss, train_dice, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE)
        val_loss, val_dice, val_iou = validate_one_epoch(model, val_loader, criterion, DEVICE)

        scheduler.step()

        print(f"  Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Train IoU: {train_iou:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}, Val Dice:   {val_dice:.4f}, Val IoU:   {val_iou:.4f}")

        # Save model state for every epoch to Drive
        epoch_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{current_epoch_display}.pth")
        torch.save(model.state_dict(), epoch_checkpoint_path)
        print(f"  Saved epoch checkpoint to {epoch_checkpoint_path}")

        # Early stopping and model saving for best model
        if val_dice > current_best_val_dice:
            current_best_val_dice = val_dice
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path_gdrive) # Save to the drive path
            print(f"  Validation Dice improved. Saving best model to {best_model_path_gdrive}")
        else:
            epochs_no_improve += 1
            print(f"  Validation Dice did not improve. Epochs with no improvement: {epochs_no_improve}")
            if epochs_no_improve >= PATIENCE:
                print(f"  Early stopping triggered after {PATIENCE} epochs without improvement.")
                break

    print("\nTraining complete!")
    print(f"Best Validation Dice (reported by training loop): {current_best_val_dice:.4f}")

if __name__ == '__main__':
    # Call main with the requested additional epochs
    main(additional_epochs=config.EPOCHS)

import os
import datetime

train_file_path = 'train.py'
print(f"\n--- Status of {train_file_path} ---")
if os.path.exists(train_file_path):
    print(f"Exists: True")
    print(f"Absolute Path: {os.path.abspath(train_file_path)}")
    print(f"Size: {os.path.getsize(train_file_path) / (1024 * 1024):.4f} MB")
    last_modified_timestamp = os.path.getmtime(train_file_path)
    print(f"Last Modified: {datetime.datetime.fromtimestamp(last_modified_timestamp)}")
else:
    print(f"Exists: False")
