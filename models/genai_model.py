import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=8): # 4 channels Target + 4 channels Input/Generated
        super().__init__()
        def block(in_c, out_c, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.model = nn.Sequential(
            block(in_channels, 64),
            block(64, 128),
            block(128, 256),
            nn.Conv2d(256, 1, 4, padding=1)
        )

    def forward(self, x, y):
        # Concatenate target and generated image for conditional discrimination
        return self.model(torch.cat([x, y], dim=1))
