
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip_connection = self.conv(x)
        out = self.pool(skip_connection)
        return out, skip_connection

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_up, x_skip):
        x_up = self.up(x_up)

        # Adjust dimensions if necessary (e.g., cropping)
        diffY = x_skip.size()[2] - x_up.size()[2]
        diffX = x_skip.size()[3] - x_up.size()[3]
        x_up = F.pad(x_up, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

        x = torch.cat([x_skip, x_up], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down1 = DownBlock(in_channels, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)

        x = self.bottleneck(x)

        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        return self.out_conv(x)

class GeneratorInpainter(nn.Module):
    """Generative AI component for Cloud Reconstruction (GAN Generator)"""
    def __init__(self, in_channels=5):
        super().__init__()
        self.unet = UNet(in_channels, 4)
    def forward(self, x, mask):
        inp = torch.cat([x, mask], dim=1)
        reconstructed_bands = torch.tanh(self.unet(inp))
        output = x * (1 - mask) + reconstructed_bands * mask
        return output

import os
import datetime

model_file_path = 'model.py'
print(f"\n--- Status of {model_file_path} ---")
if os.path.exists(model_file_path):
    print(f"Exists: True")
    print(f"Absolute Path: {os.path.abspath(model_file_path)}")
    print(f"Size: {os.path.getsize(model_file_path) / (1024 * 1024):.4f} MB")
    last_modified_timestamp = os.path.getmtime(model_file_path)
    print(f"Last Modified: {datetime.datetime.fromtimestamp(last_modified_timestamp)}")
else:
    print(f"Exists: False")
