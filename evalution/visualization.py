
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from matplotlib.colors import ListedColormap

# Configuration values are imported from config.py
import importlib
import config

# Reload config to ensure latest values
importlib.reload(config)
from config import NORM_MEAN, NORM_STD, IMAGE_SIZE

def denormalize_image(image_tensor: torch.Tensor, mean: tuple, std: tuple) -> np.ndarray:
    """
    Denormalizes a preprocessed image tensor (C, H, W) back to its original value range [0, 1]
    for visualization, assuming it was normalized with the given mean and std.

    Args:
        image_tensor (torch.Tensor): The normalized image tensor (C, H, W).
        mean (tuple): The mean values used for normalization for each channel.
        std (tuple): The standard deviation values used for normalization for each channel.

    Returns:
        np.ndarray: The denormalized image as a NumPy array (H, W, C) with pixel values in [0, 1].
    """
    mean_tensor = torch.tensor(mean, device=image_tensor.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, device=image_tensor.device).view(-1, 1, 1)

    denormalized_image = image_tensor * std_tensor + mean_tensor
    denormalized_image = torch.clamp(denormalized_image, 0, 1)

    return denormalized_image.permute(1, 2, 0).cpu().numpy()

def create_segmentation_dashboard(
    original_image_tensor: torch.Tensor,
    predicted_cloud_mask: torch.Tensor,
    output_filepath: str = 'segmentation_dashboard.png',
    mean: tuple = NORM_MEAN,
    std: tuple = NORM_STD
) -> None:
    """
    Generates and saves a segmentation dashboard visualizing the original RGB image,
    the predicted binary cloud mask, and an overlay of the mask on the RGB image.

    Args:
        original_image_tensor (torch.Tensor): The preprocessed 4-channel image tensor (C, H, W).
        predicted_cloud_mask (torch.Tensor): The binary cloud mask tensor (H, W) with 0s and 1s.
        output_filepath (str): The path to save the dashboard image.
        mean (tuple): Mean values used for normalization of the original_image_tensor.
        std (tuple): Standard deviation values used for normalization of the original_image_tensor.
    """
    rgb_image_denorm = denormalize_image(original_image_tensor[:3], mean[:3], std[:3])
    cloud_mask_np = predicted_cloud_mask.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Cloud Segmentation Dashboard', fontsize=16)

    axes[0].imshow(rgb_image_denorm)
    axes[0].set_title('Original RGB Image')
    axes[0].axis('off')

    axes[1].imshow(cloud_mask_np, cmap='gray_r')
    axes[1].set_title('Predicted Cloud Mask')
    axes[1].axis('off')

    axes[2].imshow(rgb_image_denorm)
    masked_overlay = np.zeros_like(rgb_image_denorm)
    masked_overlay[cloud_mask_np == 1, 0] = 1.0
    axes[2].imshow(masked_overlay, alpha=0.4)
    axes[2].set_title('Cloud Mask Overlay (Red)')
    axes[2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filepath)
    plt.close(fig)
    print(f"Segmentation dashboard saved to {output_filepath}")

def create_ndvi_dashboard(
    red_band: torch.Tensor,
    nir_band: torch.Tensor,
    ndvi_map: torch.Tensor,
    vegetation_health_map: torch.Tensor,
    output_filepath: str = 'ndvi_dashboard.png',
    mean: tuple = NORM_MEAN,
    std: tuple = NORM_STD
) -> None:
    """
    Generates and saves an NDVI dashboard visualizing the Red band, NIR band,
    the calculated NDVI map, and the vegetation health classification map.

    Args:
        red_band (torch.Tensor): The preprocessed Red band tensor (H, W).
        nir_band (torch.Tensor): The preprocessed NIR band tensor (H, W).
        ndvi_map (torch.Tensor): The calculated NDVI map (H, W).
        vegetation_health_map (torch.Tensor): The classified vegetation health map (H, W).
        output_filepath (str): The path to save the dashboard image.
        mean (tuple): Mean values used for normalization of the original band tensors.
        std (tuple): Standard deviation values used for normalization of the original band tensors.
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle('NDVI and Vegetation Health Dashboard', fontsize=16)

    axes[0].imshow(red_band.cpu().numpy(), cmap='gray')
    axes[0].set_title('Red Band (Normalized)')
    axes[0].axis('off')

    axes[1].imshow(nir_band.cpu().numpy(), cmap='gray')
    axes[1].set_title('NIR Band (Normalized)')
    axes[1].axis('off')

    ndvi_display = ndvi_map.cpu().numpy()
    im = axes[2].imshow(ndvi_display, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[2].set_title('NDVI Map')
    axes[2].axis('off')
    fig.colorbar(im, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)

    veg_health_display = vegetation_health_map.cpu().numpy()
    cmap_veg = ListedColormap(['#FF0000', '#FFFF00', '#008000'])
    bounds_veg = [-0.5, 0.5, 1.5, 2.5]
    norm_veg = plt.cm.colors.BoundaryNorm(bounds_veg, cmap_veg.N)

    im_veg = axes[3].imshow(veg_health_display, cmap=cmap_veg, norm=norm_veg)
    axes[3].set_title('Vegetation Health (0:Non, 1:Sparse, 2:Dense)')
    axes[3].axis('off')
    cbar = fig.colorbar(im_veg, ax=axes[3], orientation='vertical', fraction=0.046, pad=0.04, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['Non-vegetation', 'Sparse vegetation', 'Dense vegetation'])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filepath)
    plt.close(fig)
    print(f"NDVI dashboard saved to {output_filepath}")


import os
import datetime

visualization_file_path = 'visualization.py'
print(f"\n--- Status of {visualization_file_path} ---")
if os.path.exists(visualization_file_path):
    print(f"Exists: True")
    print(f"Absolute Path: {os.path.abspath(visualization_file_path)}")
    print(f"Size: {os.path.getsize(visualization_file_path) / (1024 * 1024):.4f} MB")
    last_modified_timestamp = os.path.getmtime(visualization_file_path)
    print(f"Last Modified: {datetime.datetime.fromtimestamp(last_modified_timestamp)}")
else:
    print(f"Exists: False")
