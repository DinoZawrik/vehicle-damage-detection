"""
Image processing utilities for vehicle damage detection.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union
import os
import hashlib
from pathlib import Path


class ImageValidator:
    """Utility class for image validation and preprocessing."""
    
    # Supported image formats
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # Maximum file size (50MB)
    MAX_FILE_SIZE = 50 * 1024 * 1024
    
    # Minimum and maximum dimensions
    MIN_WIDTH = 64
    MIN_HEIGHT = 64
    MAX_WIDTH = 4096
    MAX_HEIGHT = 4096
    
    @classmethod
    def validate_file(cls, file_path: Union[str, Path], 
                     check_size: bool = True,
                     check_format: bool = True) -> Tuple[bool, Optional[str]]:
        """
        Validate an image file.
        
        Args:
            file_path: Path to the image file
            check_size: Whether to check file size
            check_format: Whether to check file format
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            path = Path(file_path)
            
            # Check if file exists
            if not path.exists():
                return False, f"File does not exist: {file_path}"
            
            # Check file size
            if check_size and path.stat().st_size > cls.MAX_FILE_SIZE:
                return False, f"File too large: {path.stat().st_size} bytes (max: {cls.MAX_FILE_SIZE})"
            
            # Check file extension
            if check_format and path.suffix.lower() not in cls.SUPPORTED_FORMATS:
                return False, f"Unsupported format: {path.suffix}"
            
            # Try to load the image
            try:
                image = cv2.imread(str(path))
                if image is None:
                    return False, "Could not read image file (may be corrupted)"
            except Exception as e:
                return False, f"Error reading image: {str(e)}"
            
            # Check image dimensions
            height, width = image.shape[:2]
            if width < cls.MIN_WIDTH or height < cls.MIN_HEIGHT:
                return False, f"Image too small: {width}x{height} (min: {cls.MIN_WIDTH}x{cls.MIN_HEIGHT})"
            
            if width > cls.MAX_WIDTH or height > cls.MAX_HEIGHT:
                return False, f"Image too large: {width}x{height} (max: {cls.MAX_WIDTH}x{cls.MAX_HEIGHT})"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    @classmethod
    def preprocess_image(cls, image: np.ndarray, 
                        target_size: Tuple[int, int] = (640, 480),
                        normalize: bool = True) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image as numpy array
            target_size: Target size for resizing
            normalize: Whether to normalize pixel values
            
        Returns:
            Preprocessed image
        """
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Normalize if requested
        if normalize:
            image = image.astype(np.float32) / 255.0
        
        return image
    
    @classmethod
    def get_image_hash(cls, image_path: Union[str, Path]) -> str:
        """
        Generate a hash for an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            MD5 hash of the image file
        """
        hash_md5 = hashlib.md5()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


class ImageProcessor:
    """Utility class for image processing operations."""
    
    @staticmethod
    def enhance_image(image: np.ndarray, 
                     brightness: float = 1.0,
                     contrast: float = 1.0,
                     gamma: float = 1.0) -> np.ndarray:
        """
        Enhance image quality.
        
        Args:
            image: Input image
            brightness: Brightness adjustment (1.0 = no change)
            contrast: Contrast adjustment (1.0 = no change)
            gamma: Gamma correction (1.0 = no change)
            
        Returns:
            Enhanced image
        """
        # Apply brightness and contrast
        enhanced = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness * 255)
        
        # Apply gamma correction
        if gamma != 1.0:
            # Create lookup table for gamma correction
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                             for i in np.arange(0, 256)]).astype("uint8")
            enhanced = cv2.LUT(enhanced, table)
        
        return enhanced
    
    @staticmethod
    def denoise_image(image: np.ndarray, 
                     h: float = 10.0,
                     template_window_size: int = 7,
                     search_window_size: int = 21) -> np.ndarray:
        """
        Apply denoising to reduce noise in image.
        
        Args:
            image: Input image
            h: Filter strength
            template_window_size: Template patch size
            search_window_size: Search window size
            
        Returns:
            Denoised image
        """
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, h, h, template_window_size, search_window_size)
        else:
            return cv2.fastNlMeansDenoising(image, None, h, template_window_size, search_window_size)
    
    @staticmethod
    def sharpen_image(image: np.ndarray, 
                     kernel_size: int = 3,
                     sigma: float = 1.0) -> np.ndarray:
        """
        Apply sharpening filter.
        
        Args:
            image: Input image
            kernel_size: Size of the Gaussian kernel
            sigma: Standard deviation for Gaussian kernel
            
        Returns:
            Sharpened image
        """
        # Create Gaussian kernel
        gaussian = cv2.getGaussianKernel(kernel_size, sigma)
        
        # Apply sharpening kernel
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        
        return cv2.filter2D(image, -1, kernel)
    
    @staticmethod
    def crop_to_content(image: np.ndarray, 
                       threshold: int = 10) -> np.ndarray:
        """
        Crop image to remove empty borders.
        
        Args:
            image: Input image
            threshold: Threshold for background detection
            
        Returns:
            Cropped image
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find non-background regions
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Find the largest contour (main content)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add some padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        return image[y:y+h, x:x+w]


class ImageMetadata:
    """Utility class for image metadata extraction."""
    
    @staticmethod
    def extract_metadata(image_path: Union[str, Path]) -> dict:
        """
        Extract metadata from image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing metadata
        """
        try:
            with Image.open(image_path) as img:
                metadata = {
                    'filename': Path(image_path).name,
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'width': img.width,
                    'height': img.height,
                    'has_exif': hasattr(img, '_getexif') and img._getexif() is not None
                }
                
                # Extract EXIF data if available
                if metadata['has_exif']:
                    exif = img._getexif()
                    if exif:
                        metadata['exif_data'] = {
                            'camera_make': exif.get(271),  # Camera Make
                            'camera_model': exif.get(272),  # Camera Model
                            'date_taken': exif.get(36867),  # Date/Time Original
                            'orientation': exif.get(274),   # Orientation
                            'flash': exif.get(37385),       # Flash
                            'focal_length': exif.get(37386), # Focal Length
                            'f_number': exif.get(33437),    # F-Number
                            'exposure_time': exif.get(33434), # Exposure Time
                            'iso': exif.get(34855),         # ISO Speed Ratings
                        }
                
                return metadata
                
        except Exception as e:
            return {'error': str(e), 'filename': Path(image_path).name}
    
    @staticmethod
    def get_file_info(file_path: Union[str, Path]) -> dict:
        """
        Get file system information about image.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file information
        """
        try:
            path = Path(file_path)
            stat = path.stat()
            
            return {
                'filename': path.name,
                'size_bytes': stat.st_size,
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'created': stat.st_ctime,
                'modified': stat.st_mtime,
                'accessed': stat.st_atime,
                'extension': path.suffix.lower(),
                'exists': True
            }
            
        except Exception as e:
            return {'error': str(e), 'exists': False}


def validate_and_preprocess(image_path: Union[str, Path],
                           target_size: Tuple[int, int] = (640, 480),
                           enhance: bool = False) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Complete image validation and preprocessing pipeline.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for preprocessing
        enhance: Whether to apply image enhancement
        
    Returns:
        Tuple of (processed_image, error_message)
    """
    # Validate file
    is_valid, error = ImageValidator.validate_file(image_path)
    if not is_valid:
        return None, error
    
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return None, "Could not load image"
        
        # Crop to content if image is very large
        if image.shape[0] > 2000 or image.shape[1] > 2000:
            image = ImageProcessor.crop_to_content(image)
        
        # Apply enhancements if requested
        if enhance:
            image = ImageProcessor.enhance_image(image)
            image = ImageProcessor.denoise_image(image)
        
        # Preprocess for model
        processed_image = ImageValidator.preprocess_image(image, target_size)
        
        return processed_image, None
        
    except Exception as e:
        return None, f"Processing error: {str(e)}"


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python image_utils.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Validate and preprocess
    processed_image, error = validate_and_preprocess(image_path)
    
    if error:
        print(f"Error: {error}")
    else:
        print(f"Successfully processed image: {image_path}")
        print(f"Processed image shape: {processed_image.shape}")
        
        # Extract metadata
        metadata = ImageMetadata.extract_metadata(image_path)
        print(f"Metadata: {metadata}")