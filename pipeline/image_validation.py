# image_validation.py - Satellite Image Validation & Preprocessing

"""
Image validation module to verify satellite imagery before processing.
Handles format checking, size validation, band verification, etc.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging
from PIL import Image
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SatelliteImageValidator:
    """
    Validates satellite imagery for cloud removal processing.
    Checks format, dimensions, bands, and pixel ranges.
    """
    
    # Supported formats
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'}
    
    # Expected specifications
    MIN_IMAGE_SIZE = 64
    MAX_IMAGE_SIZE = 2048
    EXPECTED_CHANNELS = [3, 4, 8]  # RGB, RGBA, or 8-band
    MIN_PIXEL_VALUE = 0
    MAX_PIXEL_VALUE = 255
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize validator with configuration."""
        self.config = config or self._default_config()
        logger.info("✅ Satellite Image Validator initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration."""
        return {
            'min_size': self.MIN_IMAGE_SIZE,
            'max_size': self.MAX_IMAGE_SIZE,
            'expected_channels': self.EXPECTED_CHANNELS,
            'min_pixel_value': self.MIN_PIXEL_VALUE,
            'max_pixel_value': self.MAX_PIXEL_VALUE
        }
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate file format and existence.
        
        Returns:
            (is_valid, message)
        """
        try:
            path = Path(file_path)
            
            # Check file exists
            if not path.exists():
                return False, f"❌ File not found: {file_path}"
            
            # Check format
            if path.suffix.lower() not in self.SUPPORTED_FORMATS:
                return False, f"❌ Unsupported format: {path.suffix}. Supported: {self.SUPPORTED_FORMATS}"
            
            # Check file size
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > 100:
                return False, f"❌ File too large: {file_size_mb:.1f}MB (max 100MB)"
            
            return True, "✅ File format valid"
        
        except Exception as e:
            return False, f"❌ File validation error: {str(e)}"
    
    def validate_image(self, image_array: np.ndarray) -> Tuple[bool, str, Dict]:
        """
        Validate image content.
        
        Returns:
            (is_valid, message, validation_report)
        """
        validation_report = {}
        
        try:
            # Check type
            if not isinstance(image_array, np.ndarray):
                return False, "❌ Input must be numpy array", {}
            
            # Check dimensions
            if image_array.ndim != 3 and image_array.ndim != 2:
                return False, f"❌ Invalid dimensions: {image_array.ndim}D. Expected 2D or 3D", {}
            
            height, width = image_array.shape[:2]
            validation_report['height'] = height
            validation_report['width'] = width
            
            # Check size
            if height < self.config['min_size'] or width < self.config['min_size']:
                return False, f"❌ Image too small: {height}x{width}. Min: {self.config['min_size']}", validation_report
            
            if height > self.config['max_size'] or width > self.config['max_size']:
                return False, f"❌ Image too large: {height}x{width}. Max: {self.config['max_size']}", validation_report
            
            # Check channels
            if image_array.ndim == 3:
                channels = image_array.shape[2]
                validation_report['channels'] = channels
                
                if channels not in self.config['expected_channels']:
                    return False, f"❌ Invalid channels: {channels}. Expected: {self.config['expected_channels']}", validation_report
            else:
                validation_report['channels'] = 1
                logger.warning("⚠️ Grayscale image detected. Will be converted to 3-channel.")
            
            # Check pixel range
            min_val = image_array.min()
            max_val = image_array.max()
            validation_report['min_pixel'] = float(min_val)
            validation_report['max_pixel'] = float(max_val)
            
            # Warn if values are outside expected range
            if min_val < self.config['min_pixel_value']:
                logger.warning(f"⚠️ Min pixel value: {min_val} (below expected {self.config['min_pixel_value']})")
            
            if max_val > self.config['max_pixel_value'] * 2:  # Allow up to 16-bit images
                logger.warning(f"⚠️ Max pixel value: {max_val} (16-bit image detected)")
            
            # Check for NaN or Inf values
            if np.isnan(image_array).any():
                return False, "❌ Image contains NaN values", validation_report
            
            if np.isinf(image_array).any():
                return False, "❌ Image contains infinite values", validation_report
            
            # Check if image is mostly empty
            if np.mean(image_array == 0) > 0.95:
                logger.warning("⚠️ Image is mostly black (might be corrupted)")
            
            return True, "✅ Image validation passed", validation_report
        
        except Exception as e:
            logger.error(f"❌ Image validation error: {str(e)}")
            return False, f"❌ Validation error: {str(e)}", validation_report
    
    def load_and_validate(self, file_path: str) -> Tuple[bool, Optional[np.ndarray], Dict]:
        """
        Load and validate image in one step.
        
        Returns:
            (is_valid, image_array, report)
        """
        # Validate file
        file_valid, file_msg = self.validate_file(file_path)
        if not file_valid:
            return False, None, {'file_error': file_msg}
        
        logger.info(file_msg)
        
        try:
            # Load image
            image = np.array(Image.open(file_path))
            logger.info(f"Loaded image: {image.shape}, dtype: {image.dtype}")
            
            # Validate content
            img_valid, img_msg, report = self.validate_image(image)
            logger.info(img_msg)
            
            return img_valid, image if img_valid else None, report
        
        except Exception as e:
            logger.error(f"❌ Failed to load image: {str(e)}")
            return False, None, {'load_error': str(e)}
    
    def standardize_image(self, image_array: np.ndarray) -> np.ndarray:
        """
        Standardize image format for processing.
        - Convert grayscale to 3-channel
        - Normalize to [0, 1] or [0, 255]
        - Ensure proper dtype
        """
        try:
            logger.info("Standardizing image format...")
            
            # Convert grayscale to 3-channel
            if image_array.ndim == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
                logger.info("✅ Converted grayscale to 3-channel")
            
            # Normalize pixel values
            if image_array.max() <= 1.0:
                # Already normalized to [0, 1]
                image_array = (image_array * 255).astype(np.uint8)
                logger.info("✅ Converted from [0, 1] to [0, 255]")
            elif image_array.dtype == np.uint16:
                # 16-bit image
                image_array = (image_array / 256).astype(np.uint8)
                logger.info("✅ Converted from 16-bit to 8-bit")
            else:
                image_array = image_array.astype(np.uint8)
            
            logger.info(f"✅ Image standardized: {image_array.shape}, {image_array.dtype}")
            return image_array
        
        except Exception as e:
            logger.error(f"❌ Error standardizing image: {str(e)}")
            raise
    
    def get_validation_report(self, image_array: np.ndarray) -> str:
        """
        Generate human-readable validation report.
        """
        is_valid, message, report = self.validate_image(image_array)
        
        report_str = f"""
        {'='*50}
        SATELLITE IMAGE VALIDATION REPORT
        {'='*50}
        
        Status: {'✅ VALID' if is_valid else '❌ INVALID'}
        Message: {message}
        
        Image Specifications:
        - Dimensions: {report.get('height', 'N/A')} x {report.get('width', 'N/A')} pixels
        - Channels: {report.get('channels', 'N/A')}
        - Pixel Range: {report.get('min_pixel', 'N/A')} - {report.get('max_pixel', 'N/A')}
        - Data Type: {image_array.dtype}
        
        Requirements:
        - Size: {self.config['min_size']}-{self.config['max_size']} pixels
        - Channels: {self.config['expected_channels']}
        - Pixel Range: {self.config['min_pixel_value']}-{self.config['max_pixel_value']}
        
        {'='*50}
        """
        
        return report_str


# Cloud coverage detector
class CloudCoverageDetector:
    """
    Estimate cloud coverage from image statistics.
    """
    
    @staticmethod
    def estimate_coverage(image_array: np.ndarray) -> float:
        """
        Estimate cloud coverage percentage based on pixel brightness.
        Clouds typically have high values in all channels.
        
        Returns:
            cloud_coverage: Estimated percentage [0, 100]
        """
        try:
            # Convert to grayscale if needed
            if len(image_array.shape) == 3:
                gray = np.mean(image_array, axis=2)
            else:
                gray = image_array
            
            # Normalize to [0, 1]
            if gray.max() > 1:
                gray = gray / 255.0
            
            # Estimate cloud coverage (high brightness = cloud)
            # Threshold at 70% brightness
            cloud_pixels = (gray > 0.7).sum()
            total_pixels = gray.size
            coverage = (cloud_pixels / total_pixels) * 100
            
            return float(coverage)
        except Exception as e:
            logger.error(f"Error estimating cloud coverage: {e}")
            return 0.0
    
    @staticmethod
    def get_coverage_assessment(coverage: float) -> str:
        """Get human-readable coverage assessment."""
        if coverage < 10:
            return "🟢 Clear (0-10%)"
        elif coverage < 30:
            return "🟡 Partially Cloudy (10-30%)"
        elif coverage < 70:
            return "🟠 Mostly Cloudy (30-70%)"
        else:
            return "🔴 Heavily Clouded (>70%)"


# Utility functions
def validate_satellite_image(image_path: str) -> Dict:
    """
    Complete validation of satellite image.
    
    Returns dict with:
        - is_valid: bool
        - message: str
        - image: np.ndarray or None
        - report: dict
        - cloud_coverage: float
    """
    validator = SatelliteImageValidator()
    
    is_valid, image, report = validator.load_and_validate(image_path)
    
    cloud_coverage = 0.0
    if is_valid and image is not None:
        cloud_coverage = CloudCoverageDetector.estimate_coverage(image)
        coverage_assessment = CloudCoverageDetector.get_coverage_assessment(cloud_coverage)
        report['cloud_coverage'] = cloud_coverage
        report['coverage_assessment'] = coverage_assessment
    
    return {
        'is_valid': is_valid,
        'message': report.get('file_error', 'Valid') if not is_valid else '✅ Valid',
        'image': image,
        'report': report,
        'cloud_coverage': cloud_coverage
    }


if __name__ == "__main__":
    logger.info("Image Validation Module - Ready for import")