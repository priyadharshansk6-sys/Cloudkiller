"""
LISS-IV Data Handler Module
Member 3: Remote Sensing + Data Engineer
Fixed and integrated version with proper error handling
"""

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.vrt import WarpedVRT
import cv2
from pathlib import Path
import json
from typing import Tuple, List, Dict, Optional
import logging
from datetime import datetime
import pandas as pd
from scipy import ndimage
try:
    import albumentations as A
except ImportError:
    A = None

import sys
try:
    # This works when running as a standard script
    current_dir = Path(__file__).resolve()
except NameError:
    # This works inside Jupyter/Colab notebooks
    current_dir = Path.cwd().resolve()

# Add the parent directory to the system path
sys.path.insert(0, str(current_dir.parent.parent))
from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LISSIVDataHandler:
    """Handle LISS-IV satellite imagery (8-band multispectral)"""
    
    # LISS-IV Band Information
    BANDS = {
        0: 'Blue (450-520 nm)',
        1: 'Green (520-590 nm)',
        2: 'Red (620-680 nm)',
        3: 'NIR (770-860 nm)',
        4: 'SWIR1 (1550-1750 nm)',
        5: 'SWIR2 (2080-2235 nm)',
        6: 'TIR1 (10400-12500 nm)',
        7: 'TIR2 (8-14 µm)'
    }
    
    VALID_RANGES = {
        'Blue': (0, 255),
        'Green': (0, 255),
        'Red': (0, 255),
        'NIR': (0, 255),
        'SWIR1': (0, 255),
        'SWIR2': (0, 255),
        'TIR1': (0, 255),
        'TIR2': (0, 255)
    }
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize LISS-IV data handler"""
        self.data_dir = Path(data_dir) if data_dir else Path(config.data.raw_data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"LISS-IV Data Handler initialized: {self.data_dir}")
    
    def read_geotiff(self, filepath: str) -> Tuple[np.ndarray, dict]:
        """Read GeoTIFF file with geospatial metadata"""
        try:
            with rasterio.open(filepath) as src:
                image = src.read()
                metadata = src.meta.copy()
                
                logger.info(f"Loaded {filepath}")
                logger.info(f"  Shape: {image.shape}")
                logger.info(f"  CRS: {metadata.get('crs', 'Unknown')}")
                
                return image, metadata
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            raise
    
    def write_geotiff(self, image: np.ndarray, filepath: str, metadata: dict) -> None:
        """Write GeoTIFF file with geospatial metadata"""
        try:
            bands, height, width = image.shape
            
            metadata.update({
                'dtype': image.dtype,
                'nodata': None,
                'width': width,
                'height': height,
                'count': bands,
            })
            
            with rasterio.open(filepath, 'w', **metadata) as dst:
                dst.write(image)
            
            logger.info(f"Saved {filepath}")
        except Exception as e:
            logger.error(f"Error writing {filepath}: {e}")
            raise
    
    def normalize_to_reflectance(self, image: np.ndarray, ml: float = 0.0001, 
                                  al: float = -0.1) -> np.ndarray:
        """Convert DN (Digital Numbers) to reflectance"""
        try:
            reflectance = ml * image.astype(np.float32) + al
            reflectance = np.clip(reflectance, 0, 1)
            
            logger.info(f"Normalized to reflectance: min={reflectance.min():.4f}, max={reflectance.max():.4f}")
            return reflectance
        except Exception as e:
            logger.error(f"Error in normalization: {e}")
            raise
    
    def apply_radiometric_correction(self, image: np.ndarray, 
                                      gain: np.ndarray, offset: np.ndarray) -> np.ndarray:
        """Apply per-band radiometric correction"""
        try:
            corrected = image.copy().astype(np.float32)
            
            for b in range(image.shape[0]):
                corrected[b] = gain[b] * image[b] + offset[b]
            
            logger.info("Applied radiometric correction")
            return corrected
        except Exception as e:
            logger.error(f"Error in radiometric correction: {e}")
            raise
    
    def generate_quality_flags(self, image: np.ndarray) -> np.ndarray:
        """Generate quality flags for each pixel"""
        try:
            flags = np.zeros((image.shape[1], image.shape[2]), dtype=np.uint8)
            
            # Saturated pixels (value = 255)
            saturated = np.any(image == 255, axis=0)
            flags[saturated] = 5
            
            logger.info(f"Generated quality flags: {np.unique(flags, return_counts=True)}")
            return flags
        except Exception as e:
            logger.error(f"Error generating quality flags: {e}")
            raise
    
    def get_metadata_template(self) -> dict:
        """Get template metadata for LISS-IV GeoTIFF"""
        return {
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': -9999,
            'width': 1024,
            'height': 1024,
            'count': 8,
            'crs': 'EPSG:4326',
            'transform': rasterio.transform.from_bounds(0, 0, 1, 1, 1024, 1024)
        }


class CloudDetectionModule:
    """Detect clouds in LISS-IV imagery"""
    
    def __init__(self):
        """Initialize cloud detection module"""
        logger.info("Cloud Detection Module initialized")
    
    def ndsi_based_detection(self, image: np.ndarray, threshold: float = 0.4) -> np.ndarray:
        """NDSI (Normalized Difference Snow Index) based cloud detection"""
        try:
            green = image[1].astype(np.float32)
            swir1 = image[4].astype(np.float32)
            
            ndsi = (green - swir1) / (green + swir1 + 1e-8)
            cloud_mask = (ndsi > threshold).astype(np.uint8)
            
            logger.info(f"NDSI Cloud Detection: {100 * cloud_mask.mean():.2f}% cloud cover")
            return cloud_mask
        except Exception as e:
            logger.error(f"Error in NDSI detection: {e}")
            raise
    
    def thermal_cloud_detection(self, image: np.ndarray, temp_threshold: float = 280) -> np.ndarray:
        """Thermal-based cloud detection using TIR1 band"""
        try:
            tir1 = image[6].astype(np.float32)
            temperature = 250 + (tir1 / 255) * 80
            cloud_mask = (temperature < temp_threshold).astype(np.uint8)
            
            logger.info(f"Thermal Cloud Detection: {100 * cloud_mask.mean():.2f}% cloud cover")
            return cloud_mask
        except Exception as e:
            logger.error(f"Error in thermal detection: {e}")
            raise
    
    def morphological_refinement(self, mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Refine cloud mask using morphological operations"""
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            refined = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
            
            logger.info(f"Morphological refinement: {100 * refined.mean():.2f}% cloud cover")
            return refined
        except Exception as e:
            logger.error(f"Error in morphological refinement: {e}")
            raise
    
    def generate_cloud_mask(self, image: np.ndarray) -> np.ndarray:
        """Generate final cloud mask using ensemble of methods"""
        try:
            mask_ndsi = self.ndsi_based_detection(image)
            mask_thermal = self.thermal_cloud_detection(image)
            ensemble_mask = ((mask_ndsi + mask_thermal) > 0).astype(np.uint8)
            refined_mask = self.morphological_refinement(ensemble_mask)
            
            logger.info(f"Final cloud mask: {100 * refined_mask.mean():.2f}% coverage")
            return refined_mask
        except Exception as e:
            logger.error(f"Error generating final cloud mask: {e}")
            raise


class TemporalDataHandler:
    """Handle temporal sequences of LISS-IV images"""
    
    def __init__(self):
        """Initialize temporal data handler"""
        logger.info("Temporal Data Handler initialized")
    
    def stack_temporal_sequence(self, images: List[np.ndarray]) -> np.ndarray:
        """Stack temporal sequence of images"""
        try:
            temporal_stack = np.stack(images, axis=0)
            
            logger.info(f"Temporal stack shape: {temporal_stack.shape}")
            logger.info(f"  {temporal_stack.shape[0]} time steps")
            logger.info(f"  {temporal_stack.shape[1]} bands")
            
            return temporal_stack
        except Exception as e:
            logger.error(f"Error stacking temporal sequence: {e}")
            raise
    
    def temporal_consistency_check(self, temporal_stack: np.ndarray, 
                                    threshold: float = 0.1) -> np.ndarray:
        """Check temporal consistency across time steps"""
        try:
            variance = np.var(temporal_stack, axis=0)
            mean_variance = np.mean(variance, axis=0)
            consistency = np.exp(-mean_variance / threshold)
            
            logger.info(f"Temporal consistency: min={consistency.min():.3f}, max={consistency.max():.3f}")
            return consistency
        except Exception as e:
            logger.error(f"Error in temporal consistency check: {e}")
            raise
    
    def gap_filling_with_temporal_interpolation(self, temporal_stack: np.ndarray,
                                                  cloud_mask: np.ndarray) -> np.ndarray:
        """Fill cloud gaps using temporal interpolation"""
        try:
            filled_stack = temporal_stack.copy()
            cloudy_pixels = np.where(cloud_mask > 0)
            
            for idx, idy in zip(cloudy_pixels[0], cloudy_pixels[1]):
                clear_obs = temporal_stack[1:, :, idx, idy]
                mean_value = np.mean(clear_obs, axis=0)
                filled_stack[0, :, idx, idy] = mean_value
            
            logger.info(f"Gap-filled {len(cloudy_pixels[0])} cloudy pixels")
            return filled_stack
        except Exception as e:
            logger.error(f"Error in gap filling: {e}")
            raise
    
    def compute_temporal_features(self, temporal_stack: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute temporal features for each spatial location"""
        try:
            features = {
                'mean': np.mean(temporal_stack, axis=0),
                'std': np.std(temporal_stack, axis=0),
                'min': np.min(temporal_stack, axis=0),
                'max': np.max(temporal_stack, axis=0),
                'range': np.max(temporal_stack, axis=0) - np.min(temporal_stack, axis=0)
            }
            
            logger.info("Computed temporal features")
            return features
        except Exception as e:
            logger.error(f"Error computing temporal features: {e}")
            raise


class SARDataHandler:
    """Handle Sentinel-1 SAR data integration"""
    
    def __init__(self):
        """Initialize SAR data handler"""
        logger.info("SAR Data Handler initialized")
        self.polarizations = ['VV', 'VH']
        self.resolution = 10  # meters
    
    def read_sentinel1(self, vv_path: str, vh_path: str) -> np.ndarray:
        """Read Sentinel-1 VV and VH polarizations"""
        try:
            with rasterio.open(vv_path) as src:
                vv = src.read(1).astype(np.float32)
            
            with rasterio.open(vh_path) as src:
                vh = src.read(1).astype(np.float32)
            
            sar_image = np.stack([vv, vh], axis=0)
            
            logger.info(f"Loaded SAR image: {sar_image.shape}")
            logger.info(f"  VV range: {vv.min():.2f} - {vv.max():.2f}")
            logger.info(f"  VH range: {vh.min():.2f} - {vh.max():.2f}")
            
            return sar_image
        except Exception as e:
            logger.error(f"Error reading SAR data: {e}")
            raise
    
    def speckle_filtering(self, sar_image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Apply speckle filtering to SAR image"""
        try:
            filtered = sar_image.copy()
            
            for i in range(sar_image.shape[0]):
                filtered[i] = cv2.medianBlur(sar_image[i].astype(np.uint8), kernel_size)
            
            logger.info(f"Applied speckle filtering (kernel={kernel_size})")
            return filtered
        except Exception as e:
            logger.error(f"Error in speckle filtering: {e}")
            raise
    
    def compute_sar_features(self, sar_image: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute SAR-derived features"""
        try:
            vv = sar_image[0]
            vh = sar_image[1]
            ratio = (vv + 1e-8) / (vh + 1e-8)
            ratio = np.clip(ratio, 0, 10)
            
            edges_vv = cv2.Sobel(vv.astype(np.float32), cv2.CV_32F, 1, 1, ksize=3)
            edges_vh = cv2.Sobel(vh.astype(np.float32), cv2.CV_32F, 1, 1, ksize=3)
            
            features = {
                'vv': vv,
                'vh': vh,
                'ratio': ratio,
                'edges': np.stack([edges_vv, edges_vh], axis=0)
            }
            
            logger.info("Computed SAR features")
            return features
        except Exception as e:
            logger.error(f"Error computing SAR features: {e}")
            raise
    
    def resample_to_liss_resolution(self, sar_image: np.ndarray, 
                                     scale_factor: float = 3.0) -> np.ndarray:
        """Resample SAR (10m) to LISS-IV (30m) resolution"""
        try:
            h, w = sar_image.shape[1:]
            new_h = int(h / scale_factor)
            new_w = int(w / scale_factor)
            
            resampled = np.zeros((2, new_h, new_w), dtype=np.float32)
            
            for i in range(2):
                resampled[i] = cv2.resize(sar_image[i], (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            logger.info(f"Resampled SAR from {sar_image.shape[1:]} to {resampled.shape[1:]}")
            return resampled
        except Exception as e:
            logger.error(f"Error resampling SAR: {e}")
            raise


class DataAugmentationModule:
    """Data augmentation for training"""
    
    def __init__(self):
        """Initialize augmentation module"""
        if A is None:
            logger.warning("albumentations not installed. Augmentation disabled.")
            self.augmenter = None
        else:
            self.augmenter = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=45, p=0.5),
                A.ElasticTransform(p=0.3),
                A.GaussNoise(p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.2),
            ], p=0.8)
        
        logger.info("Data Augmentation Module initialized")
    
    def augment_image_pair(self, cloudy: np.ndarray, clear: np.ndarray,
                           cloud_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Augment cloudy-clear image pair consistently"""
        if self.augmenter is None:
            logger.warning("Augmenter not available, returning original images")
            return cloudy, clear, cloud_mask
        
        try:
            cloudy_hw = np.transpose(cloudy, (1, 2, 0))
            clear_hw = np.transpose(clear, (1, 2, 0))
            
            augmented_cloudy = self.augmenter(image=cloudy_hw)['image']
            augmented_clear = self.augmenter(image=clear_hw)['image']
            augmented_mask = self.augmenter(image=cloud_mask)['image']
            
            augmented_cloudy = np.transpose(augmented_cloudy, (2, 0, 1))
            augmented_clear = np.transpose(augmented_clear, (2, 0, 1))
            
            return augmented_cloudy, augmented_clear, augmented_mask
        except Exception as e:
            logger.error(f"Error in augmentation: {e}")
            return cloudy, clear, cloud_mask


class DataPipeline:
    """End-to-end data pipeline"""
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize complete pipeline"""
        self.data_dir = Path(data_dir) if data_dir else Path(config.data.raw_data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.liss = LISSIVDataHandler(str(self.data_dir))
        self.cloud = CloudDetectionModule()
        self.temporal = TemporalDataHandler()
        self.sar = SARDataHandler()
        self.augment = DataAugmentationModule()
        
        logger.info("Complete Data Pipeline initialized")
    
    def process_single_scene(self, cloudy_path: str, clear_path: str,
                             sar_vv_path: Optional[str] = None,
                             sar_vh_path: Optional[str] = None,
                             temporal_paths: Optional[List[str]] = None,
                             output_dir: Optional[str] = None) -> Dict:
        """Process a single cloudy-clear scene pair"""
        try:
            logger.info(f"Processing: {cloudy_path}")
            
            output_dir = Path(output_dir) if output_dir else Path(config.data.processed_data_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load images
            cloudy, cloudy_meta = self.liss.read_geotiff(cloudy_path)
            clear, clear_meta = self.liss.read_geotiff(clear_path)
            
            # Normalize to reflectance
            cloudy_reflectance = self.liss.normalize_to_reflectance(cloudy)
            clear_reflectance = self.liss.normalize_to_reflectance(clear)
            
            # Generate cloud mask
            cloud_mask = self.cloud.generate_cloud_mask(cloudy_reflectance)
            
            # Load temporal data if provided
            temporal_stack = None
            if temporal_paths:
                temporal_images = [cloudy_reflectance]
                for path in temporal_paths:
                    img, _ = self.liss.read_geotiff(path)
                    img_ref = self.liss.normalize_to_reflectance(img)
                    temporal_images.append(img_ref)
                temporal_stack = self.temporal.stack_temporal_sequence(temporal_images)
            
            # Load SAR if provided
            sar_image = None
            if sar_vv_path and sar_vh_path:
                sar_image = self.sar.read_sentinel1(sar_vv_path, sar_vh_path)
                sar_image = self.sar.speckle_filtering(sar_image)
                sar_image = self.sar.resample_to_liss_resolution(sar_image)
            
            # Save processed data
            result = {
                'cloudy': cloudy_reflectance,
                'clear': clear_reflectance,
                'cloud_mask': cloud_mask,
                'temporal_stack': temporal_stack,
                'sar_image': sar_image,
                'metadata': cloudy_meta
            }
            
            # Save to disk
            np.save(str(output_dir / 'cloudy.npy'), cloudy_reflectance)
            np.save(str(output_dir / 'clear.npy'), clear_reflectance)
            np.save(str(output_dir / 'cloud_mask.npy'), cloud_mask)
            
            if temporal_stack is not None:
                np.save(str(output_dir / 'temporal_stack.npy'), temporal_stack)
            
            if sar_image is not None:
                np.save(str(output_dir / 'sar_image.npy'), sar_image)
            
            # Save metadata
            with open(output_dir / 'metadata.json', 'w') as f:
                json.dump({
                    'shape_cloudy': cloudy_reflectance.shape,
                    'shape_clear': clear_reflectance.shape,
                    'cloud_coverage': float(cloud_mask.mean()),
                    'temporal_stack_available': temporal_stack is not None,
                    'sar_available': sar_image is not None
                }, f, indent=2)
            
            logger.info(f"Saved processed data to {output_dir}")
            return result
        
        except Exception as e:
            logger.error(f"Error processing scene: {e}")
            raise


if __name__ == "__main__":
    logger.info("Data Pipeline Module - Ready for import")