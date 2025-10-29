"""
Custom validators and error handling for the Vehicle Damage Detection API.
"""

from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import re
import mimetypes
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, error_code: Optional[str] = None):
        self.message = message
        self.field = field
        self.error_code = error_code or "VALIDATION_ERROR"
        super().__init__(self.message)


class FileValidationError(ValidationError):
    """Exception for file-related validation errors."""
    
    def __init__(self, message: str, file_path: Optional[str] = None):
        super().__init__(message, error_code="FILE_VALIDATION_ERROR")
        self.file_path = file_path


class RequestValidationError(ValidationError):
    """Exception for request validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        super().__init__(message, field, "REQUEST_VALIDATION_ERROR")
        self.value = value


class ImageValidator:
    """Enhanced image validation with comprehensive checks."""
    
    # Supported image formats with MIME types
    SUPPORTED_FORMATS = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.bmp': 'image/bmp',
        '.tiff': 'image/tiff',
        '.webp': 'image/webp'
    }
    
    # File size limits (in bytes)
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MIN_FILE_SIZE = 1024  # 1KB
    
    # Dimension limits
    MIN_WIDTH = 64
    MIN_HEIGHT = 64
    MAX_WIDTH = 4096
    MAX_HEIGHT = 4096
    
    # Aspect ratio limits (to prevent extremely wide/tall images)
    MIN_ASPECT_RATIO = 0.1
    MAX_ASPECT_RATIO = 10.0
    
    @classmethod
    def validate_image_file(cls, file_data: bytes, filename: str, 
                          content_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive image file validation.
        
        Args:
            file_data: Raw file data as bytes
            filename: Original filename
            content_type: MIME type from request
            
        Returns:
            Dictionary with validation results
            
        Raises:
            FileValidationError: If validation fails
        """
        validation_result = {
            "valid": False,
            "warnings": [],
            "errors": [],
            "metadata": {}
        }
        
        try:
            # 1. Basic file checks
            cls._validate_basic_file_properties(file_data, filename, validation_result)
            
            # 2. Format validation
            cls._validate_image_format(file_data, filename, content_type, validation_result)
            
            # 3. Image data validation
            cls._validate_image_data(file_data, validation_result)
            
            # 4. Image properties validation
            cls._validate_image_properties(file_data, validation_result)
            
            # 5. Security checks
            cls._validate_security(file_data, filename, validation_result)
            
            # Set final validation status
            validation_result["valid"] = len(validation_result["errors"]) == 0
            
            if not validation_result["valid"]:
                raise FileValidationError(
                    f"Image validation failed: {'; '.join(validation_result['errors'])}",
                    file_path=filename
                )
            
            return validation_result
            
        except FileValidationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during image validation: {e}")
            raise FileValidationError(f"Unexpected validation error: {str(e)}", file_path=filename)
    
    @classmethod
    def _validate_basic_file_properties(cls, file_data: bytes, filename: str, 
                                      result: Dict[str, Any]) -> None:
        """Validate basic file properties."""
        
        # Check file size
        if len(file_data) < cls.MIN_FILE_SIZE:
            result["errors"].append(f"File too small: {len(file_data)} bytes (min: {cls.MIN_FILE_SIZE})")
        
        if len(file_data) > cls.MAX_FILE_SIZE:
            result["errors"].append(f"File too large: {len(file_data)} bytes (max: {cls.MAX_FILE_SIZE})")
        
        # Check filename
        if not filename:
            result["errors"].append("Missing filename")
            return
        
        # Check for suspicious filenames
        if cls._is_suspicious_filename(filename):
            result["warnings"].append("Suspicious filename detected")
        
        # Check file extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in cls.SUPPORTED_FORMATS:
            result["errors"].append(f"Unsupported file format: {file_ext}")
    
    @classmethod
    def _validate_image_format(cls, file_data: bytes, filename: str, 
                             content_type: Optional[str], result: Dict[str, Any]) -> None:
        """Validate image format and MIME type."""
        
        # Verify MIME type matches extension
        if content_type:
            expected_mime = cls.SUPPORTED_FORMATS.get(Path(filename).suffix.lower())
            if expected_mime and content_type != expected_mime:
                result["warnings"].append(
                    f"MIME type mismatch: declared {content_type}, expected {expected_mime}"
                )
        
        # Try to detect actual format
        try:
            with Image.open(file_data) as img:
                actual_format = img.format
                if actual_format and actual_format.upper() not in ['JPEG', 'PNG', 'BMP', 'TIFF', 'WEBP']:
                    result["errors"].append(f"Unsupported image format: {actual_format}")
        except Exception:
            result["errors"].append("Could not parse image format")
    
    @classmethod
    def _validate_image_data(cls, file_data: bytes, result: Dict[str, Any]) -> None:
        """Validate image data integrity."""
        
        try:
            with Image.open(file_data) as img:
                # Get basic metadata
                result["metadata"] = {
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "has_transparency": img.mode in ('RGBA', 'LA', 'P')
                }
                
                # Check image dimensions
                width, height = img.size
                
                if width < cls.MIN_WIDTH:
                    result["errors"].append(f"Image too narrow: {width}px (min: {cls.MIN_WIDTH}px)")
                
                if height < cls.MIN_HEIGHT:
                    result["errors"].append(f"Image too short: {height}px (min: {cls.MIN_HEIGHT}px)")
                
                if width > cls.MAX_WIDTH:
                    result["warnings"].append(f"Image very wide: {width}px (max recommended: {cls.MAX_WIDTH}px)")
                
                if height > cls.MAX_HEIGHT:
                    result["warnings"].append(f"Image very tall: {height}px (max recommended: {cls.MAX_HEIGHT}px)")
                
                # Check aspect ratio
                aspect_ratio = width / height
                if aspect_ratio < cls.MIN_ASPECT_RATIO:
                    result["warnings"].append(f"Unusual aspect ratio: {aspect_ratio:.2f} (very tall)")
                elif aspect_ratio > cls.MAX_ASPECT_RATIO:
                    result["warnings"].append(f"Unusual aspect ratio: {aspect_ratio:.2f} (very wide)")
                
                # Check for very small files (possible corruption)
                file_size_per_pixel = len(file_data) / (width * height)
                if file_size_per_pixel < 0.1:  # Less than 0.1 bytes per pixel
                    result["warnings"].append("Image may be heavily compressed or corrupted")
                
        except Exception as e:
            result["errors"].append(f"Could not validate image data: {str(e)}")
    
    @classmethod
    def _validate_image_properties(cls, file_data: bytes, result: Dict[str, Any]) -> None:
        """Validate image properties and quality."""
        
        try:
            with Image.open(file_data) as img:
                width, height = img.size
                
                # Check for very large images that might cause memory issues
                total_pixels = width * height
                if total_pixels > 16_000_000:  # 16MP
                    result["warnings"].append(f"Large image: {total_pixels:,} pixels (may be slow to process)")
                
                # Check color mode
                if img.mode not in ['RGB', 'RGBA', 'L', 'P']:
                    result["warnings"].append(f"Unusual color mode: {img.mode}")
                
                # Check for animated images (not supported)
                if hasattr(img, 'is_animated') and img.is_animated:
                    result["errors"].append("Animated images are not supported")
                
        except Exception:
            # Image properties validation failures are warnings, not errors
            result["warnings"].append("Could not validate some image properties")
    
    @classmethod
    def _validate_security(cls, file_data: bytes, filename: str, result: Dict[str, Any]) -> None:
        """Basic security validation."""
        
        # Check for potential malicious content
        try:
            # Check file header for common image formats
            header = file_data[:16].hex()
            
            # JPEG: FFD8
            # PNG: 89504E47
            # BMP: 424D
            # TIFF: 49492A00 or 4D4D002A
            # WEBP: 52494646
            
            valid_headers = ['ffd8', '89504e47', '424d', '49492a00', '4d4d002a', '52494646']
            
            if not any(header.lower().startswith(h) for h in valid_headers):
                result["errors"].append("Invalid image file header (possible corruption or malicious content)")
            
        except Exception:
            result["warnings"].append("Could not perform security checks")
    
    @classmethod
    def _is_suspicious_filename(cls, filename: str) -> bool:
        """Check if filename appears suspicious."""
        
        suspicious_patterns = [
            r'\.\.',  # Path traversal
            r'[<>:"|?*]',  # Invalid Windows characters
            r'^(con|prn|aux|nul|com[1-9]|lpt[1-9])$',  # Windows reserved names
            r'\.(exe|bat|cmd|scr|pif|vbs|js)$',  # Executable extensions
            r'\0',  # Null bytes
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, filename, re.IGNORECASE):
                return True
        
        return False


class RequestValidator:
    """Validator for API request parameters."""
    
    @staticmethod
    def validate_client_id(client_id: Optional[str]) -> Optional[str]:
        """Validate client ID parameter."""
        if client_id is None:
            return None
        
        if not isinstance(client_id, str):
            raise RequestValidationError("client_id must be a string", field="client_id", value=client_id)
        
        if len(client_id) > 100:
            raise RequestValidationError("client_id too long (max 100 characters)", field="client_id", value=client_id)
        
        if not re.match(r'^[a-zA-Z0-9_-]+$', client_id):
            raise RequestValidationError("client_id contains invalid characters", field="client_id", value=client_id)
        
        return client_id
    
    @staticmethod
    def validate_session_id(session_id: Optional[str]) -> Optional[str]:
        """Validate session ID parameter."""
        if session_id is None:
            return None
        
        if not isinstance(session_id, str):
            raise RequestValidationError("session_id must be a string", field="session_id", value=session_id)
        
        if len(session_id) > 100:
            raise RequestValidationError("session_id too long (max 100 characters)", field="session_id", value=session_id)
        
        if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
            raise RequestValidationError("session_id contains invalid characters", field="session_id", value=session_id)
        
        return session_id
    
    @staticmethod
    def validate_analysis_options(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate analysis options."""
        if options is None:
            return {}
        
        validated_options = {}
        
        # Confidence threshold
        if 'confidence_threshold' in options:
            conf = options['confidence_threshold']
            if not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
                raise RequestValidationError(
                    "confidence_threshold must be between 0 and 1",
                    field="confidence_threshold",
                    value=conf
                )
            validated_options['confidence_threshold'] = float(conf)
        
        # Visualization options
        if 'visualize' in options:
            vis = options['visualize']
            if not isinstance(vis, bool):
                raise RequestValidationError(
                    "visualize must be a boolean",
                    field="visualize",
                    value=vis
                )
            validated_options['visualize'] = vis
        
        # Custom parameters can be added here
        for key, value in options.items():
            if key not in ['confidence_threshold', 'visualize']:
                validated_options[key] = value
        
        return validated_options


class ErrorHandler:
    """Centralized error handling and response formatting."""
    
    @staticmethod
    def format_validation_error(error: ValidationError) -> Dict[str, Any]:
        """Format validation error for API response."""
        return {
            "error": "validation_error",
            "message": error.message,
            "details": {
                "field": error.field,
                "error_code": error.error_code
            },
            "status_code": 422
        }
    
    @staticmethod
    def format_file_validation_error(error: FileValidationError) -> Dict[str, Any]:
        """Format file validation error for API response."""
        return {
            "error": "file_validation_error",
            "message": error.message,
            "details": {
                "file_path": error.file_path,
                "error_code": error.error_code
            },
            "status_code": 400
        }
    
    @staticmethod
    def format_request_validation_error(error: RequestValidationError) -> Dict[str, Any]:
        """Format request validation error for API response."""
        return {
            "error": "request_validation_error",
            "message": error.message,
            "details": {
                "field": error.field,
                "value": error.value,
                "error_code": error.error_code
            },
            "status_code": 422
        }
    
    @staticmethod
    def handle_unexpected_error(error: Exception) -> Dict[str, Any]:
        """Handle unexpected errors."""
        logger.error(f"Unexpected error: {str(error)}", exc_info=True)
        
        return {
            "error": "internal_server_error",
            "message": "An unexpected error occurred",
            "details": {
                "error_type": type(error).__name__
            },
            "status_code": 500
        }


def validate_image_upload(file_data: bytes, filename: str, 
                         client_id: Optional[str] = None,
                         session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Complete image upload validation.
    
    Args:
        file_data: Raw file data
        filename: Original filename
        client_id: Optional client identifier
        session_id: Optional session identifier
        
    Returns:
        Validation results dictionary
        
    Raises:
        FileValidationError: If file validation fails
        RequestValidationError: If request validation fails
    """
    # Validate file
    file_validation = ImageValidator.validate_image_file(file_data, filename)
    
    # Validate request parameters
    validated_client_id = RequestValidator.validate_client_id(client_id)
    validated_session_id = RequestValidator.validate_session_id(session_id)
    
    return {
        "file_valid": file_validation["valid"],
        "file_metadata": file_validation["metadata"],
        "validation_warnings": file_validation["warnings"],
        "validation_errors": file_validation["errors"],
        "client_id": validated_client_id,
        "session_id": validated_session_id
    }


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python validators.py <image_file>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        with open(image_path, "rb") as f:
            file_data = f.read()
        
        result = validate_image_upload(file_data, image_path)
        
        print(f"Validation result for {image_path}:")
        print(f"Valid: {result['file_valid']}")
        print(f"Metadata: {result['file_metadata']}")
        print(f"Warnings: {result['validation_warnings']}")
        print(f"Errors: {result['validation_errors']}")
        
    except Exception as e:
        print(f"Validation failed: {e}")