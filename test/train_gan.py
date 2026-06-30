import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import config
from model import UNet
from genai_model import PatchGANDiscriminator
from reconstruction_dataset_v2 import ReconstructionDatasetV2
from metrics import calculate_sam

def save_progress_image(epoch, generator, val_loader, device, path):
    generator.eval()
    with torch.no_grad():
        inputs, targets = next(iter(val_loader))
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = generator(inputs)

        # Denorm for visualization (RGB only)
        def to_rgb(t):
            m = torch.tensor(config.NORM_MEAN[:3]).view(1,3,1,1).to(device)
            s = torch.tensor(config.NORM_STD[:3]).view(1,3,1,1).to(device)
            res = (t[:, :3, :, :] * s + m).clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)
            return res[0]

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(to_rgb(inputs[:, :4, :, :]))
        ax[0].set_title("Cloudy Input")
        ax[1].imshow(to_rgb(outputs))
        ax[1].set_title("GAN Reconstruction")
        ax[2].imshow(to_rgb(targets))
        ax[2].set_title("Ground Truth")
        for a in ax: a.axis('off')
        plt.savefig(path)
        plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CHECKPOINT_DIR = "/content/drive/MyDrive/CloudKillers_Checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Hyperparameters
    BATCH_SIZE = 4
    LR = 0.0002
    LAMBDA_L1 = 100
    EPOCHS = 50

    # Data
    full_ds = ReconstructionDatasetV2()
    train_n = int(0.8 * len(full_ds))
    val_n = len(full_ds) - train_n
    train_ds, val_ds = random_split(full_ds, [train_n, val_n], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Models (Generator outputting 4 channels for RGB+NIR)
    netG = UNet(in_channels=5, out_channels=4).to(device)
    netD = PatchGANDiscriminator(in_channels=8).to(device) # Input(4) + Target(4)

    optG = optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.999))

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    scaler = GradScaler()

    history = []

    for epoch in range(EPOCHS):
        netG.train()
        netD.train()
        g_running, d_running = 0.0, 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)

            # Train Discriminator
            optD.zero_grad()
            with autocast():
                fake = netG(inputs)
                pred_real = netD(inputs[:, :4, :, :], targets)
                loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
                pred_fake = netD(inputs[:, :4, :, :], fake.detach())
                loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
                loss_D = (loss_D_real + loss_D_fake) * 0.5

            scaler.scale(loss_D).backward()
            scaler.step(optD)

            # Train Generator
            optG.zero_grad()
            with autocast():
                pred_fake = netD(inputs[:, :4, :, :], fake)
                loss_G_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
                loss_G_L1 = criterion_L1(fake, targets) * LAMBDA_L1
                loss_G = loss_G_GAN + loss_G_L1

            scaler.scale(loss_G).backward()
            scaler.step(optG)
            scaler.update()

            g_running += loss_G.item()
            d_running += loss_D.item()
            loop.set_postfix(G_loss=loss_G.item(), D_loss=loss_D.item())

        # Validation
        netG.eval()
        val_l1, val_sam = 0.0, 0.0
        with torch.no_grad():
            for v_in, v_tar in val_loader:
                v_in, v_tar = v_in.to(device), v_tar.to(device)
                v_fake = netG(v_in)
                val_l1 += criterion_L1(v_fake, v_tar).item()
                val_sam += calculate_sam(v_fake[0], v_tar[0]).item()

        avg_val_l1 = val_l1 / len(val_loader)
        avg_val_sam = val_sam / len(val_loader)

        print(f"Summary -> G_Loss: {g_running/len(train_loader):.4f} | D_Loss: {d_running/len(train_loader):.4f} | Val L1: {avg_val_l1:.4f} | Val SAM: {avg_val_sam:.4f}")

        history.append({'epoch': epoch+1, 'g_loss': g_running/len(train_loader), 'd_loss': d_running/len(train_loader), 'val_l1': avg_val_l1, 'val_sam': avg_val_sam})

        # Checkpointing
        torch.save(netG.state_dict(), os.path.join(CHECKPOINT_DIR, 'latest_generator.pth'))
        if (epoch + 1) % 5 == 0:
            save_progress_image(epoch+1, netG, val_loader, device, f"gan_progress_epoch_{epoch+1}.png")

    pd.DataFrame(history).to_csv("gan_metrics.csv", index=False)
    torch.save(netG.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_generator.pth'))
    torch.save(netD.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_discriminator.pth'))

if __name__ == '__main__':
    main()

import os
import datetime

train_gan_file_path = 'train_gan.py'
print(f"\n--- Status of {train_gan_file_path} ---")
if os.path.exists(train_gan_file_path):
    print(f"Exists: True")
    print(f"Absolute Path: {os.path.abspath(train_gan_file_path)}")
    print(f"Size: {os.path.getsize(train_gan_file_path) / (1024 * 1024):.4f} MB")
    last_modified_timestamp = os.path.getmtime(train_gan_file_path)
    print(f"Last Modified: {datetime.datetime.fromtimestamp(last_modified_timestamp)}")
else:
    print(f"Exists: False")
