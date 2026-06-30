"""
Configuration Module - CloudKiller
Central configuration management for all project settings
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Project Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = PROJECT_ROOT / "models"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
SAMPLE_TEST_DIR = PROJECT_ROOT / "sample_test"

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, MODEL_DIR, EVALUATION_DIR, SAMPLE_TEST_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Data Pipeline Configuration"""
    raw_data_dir: str = str(DATA_DIR / "raw")
    processed_data_dir: str = str(DATA_DIR / "processed")
    temporal_data_dir: str = str(DATA_DIR / "temporal")
    
    # LISS-IV Specifications
    num_bands: int = 8
    image_resolution: int = 30  # meters
    
    # Band Information
    band_names: list = None
    
    def __post_init__(self):
        if self.band_names is None:
            self.band_names = [
                'Blue (450-520 nm)',
                'Green (520-590 nm)',
                'Red (620-680 nm)',
                'NIR (770-860 nm)',
                'SWIR1 (1550-1750 nm)',
                'SWIR2 (2080-2235 nm)',
                'TIR1 (10400-12500 nm)',
                'TIR2 (8-14 µm)'
            ]
    
    def __str__(self):
        return f"""
        Data Configuration:
        - Raw Data: {self.raw_data_dir}
        - Processed Data: {self.processed_data_dir}
        - Temporal Data: {self.temporal_data_dir}
        - Bands: {self.num_bands}
        - Resolution: {self.image_resolution}m
        """


@dataclass
class ModelConfig:
    """Model Architecture Configuration"""
    # Cloud Segmentation
    segmentation_model: str = "FCN-ResNet50"
    num_classes: int = 2  # Cloud/Non-cloud
    
    # Diffusion Model
    timesteps: int = 1000
    num_inference_steps: int = 50
    beta_start: float = 0.0001
    beta_end: float = 0.02
    
    # Temporal Attention
    num_temporal_frames: int = 4  # T-1, T-2, T-3, plus current
    temporal_feature_dim: int = 512
    num_attention_heads: int = 8
    
    # SAR Integration
    sar_feature_dim: int = 256
    use_sar_guidance: bool = True
    
    # Model Paths
    segmentation_weights: str = str(MODEL_DIR / "segmentation_model.pth")
    diffusion_weights: str = str(MODEL_DIR / "diffusion_model.pth")
    
    def __str__(self):
        return f"""
        Model Configuration:
        - Segmentation: {self.segmentation_model}
        - Timesteps: {self.timesteps}
        - Inference Steps: {self.num_inference_steps}
        - Temporal Frames: {self.num_temporal_frames}
        - SAR Guidance: {self.use_sar_guidance}
        """


@dataclass
class InferenceConfig:
    """Inference Configuration"""
    device: str = "cuda"  # cuda or cpu
    batch_size: int = 4
    confidence_threshold: float = 0.75
    use_sar: bool = True
    use_temporal: bool = True
    output_format: str = "geotiff"  # geotiff or numpy
    
    # Processing
    save_intermediate: bool = True
    save_cloud_mask: bool = True
    save_ndvi_map: bool = True
    
    def __str__(self):
        return f"""
        Inference Configuration:
        - Device: {self.device}
        - Batch Size: {self.batch_size}
        - Confidence Threshold: {self.confidence_threshold}
        - SAR: {self.use_sar}
        - Temporal: {self.use_temporal}
        """


@dataclass
class ValidationConfig:
    """Validation & Metrics Configuration"""
    # Spectral Metrics Thresholds
    sam_threshold: float = 0.1  # Spectral Angle Mapper (radians)
    ssim_threshold: float = 0.75  # Structural Similarity
    psnr_threshold: float = 25.0  # Peak Signal-to-Noise Ratio (dB)
    
    # Agricultural Validation
    ndvi_healthy_threshold: float = 0.6
    ndvi_moderate_threshold: float = 0.4
    ndvi_anomaly_threshold: float = 0.15
    
    # Overall Confidence
    min_confidence_score: float = 0.75
    
    # Report Settings
    generate_report: bool = True
    report_format: str = "pdf"  # pdf or html
    save_visualizations: bool = True
    
    def __str__(self):
        return f"""
        Validation Configuration:
        - SAM Threshold: {self.sam_threshold}
        - SSIM Threshold: {self.ssim_threshold}
        - PSNR Threshold: {self.psnr_threshold}
        - NDVI Healthy: {self.ndvi_healthy_threshold}
        """


@dataclass
class UIConfig:
    """Streamlit UI Configuration"""
    app_title: str = "CloudKiller - Cloud Removal for LISS-IV"
    app_icon: str = "🛰️"
    page_layout: str = "wide"
    theme: str = "light"
    
    # UI Features
    enable_sar_upload: bool = True
    enable_temporal_upload: bool = True
    enable_download: bool = True
    enable_api: bool = True
    
    # File Upload Limits
    max_file_size_mb: int = 100
    allowed_formats: list = None
    
    def __post_init__(self):
        if self.allowed_formats is None:
            self.allowed_formats = ["tif", "tiff", "TIF", "TIFF"]


# Global Configuration Instance
class Config:
    """Master Configuration Class"""
    
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.inference = InferenceConfig()
        self.validation = ValidationConfig()
        self.ui = UIConfig()
    
    def __str__(self):
        return f"""
        {'='*60}
        CLOUDKILLER PROJECT CONFIGURATION
        {'='*60}
        {self.data}
        {self.model}
        {self.inference}
        {self.validation}
        {'='*60}
        """


# Singleton instance
config = Config()

if __name__ == "__main__":
    print(config)