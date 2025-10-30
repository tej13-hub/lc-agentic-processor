"""
Image preprocessing for document enhancement
"""

import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocesses document images for OCR."""
    
    def __init__(self, target_dpi=300):
        self.target_dpi = target_dpi
        logger.info("ImagePreprocessor initialized")
    
    def pil_to_cv2(self, pil_image):
        """Convert PIL to OpenCV format."""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def cv2_to_pil(self, cv2_image):
        """Convert OpenCV to PIL format."""
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
    
    def convert_to_grayscale(self, image):
        """Convert to grayscale."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            logger.info("Converted to grayscale")
            return gray
        return image
    
    def remove_noise(self, image):
        """Remove noise using Non-Local Means Denoising."""
        denoised = cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)
        logger.info("Noise removed")
        return denoised
    
    def enhance_contrast(self, image, method='clahe'):
        """
        Enhance contrast using various methods.
        
        Args:
            image: Grayscale image
            method: 'clahe', 'histogram', or 'adaptive'
        """
        if method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        elif method == 'histogram':
            enhanced = cv2.equalizeHist(image)
        elif method == 'adaptive':
            from skimage import exposure
            enhanced = exposure.equalize_adapthist(image, clip_limit=0.03)
            enhanced = (enhanced * 255).astype(np.uint8)
        else:
            logger.warning(f"Unknown method: {method}, using clahe")
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        logger.info(f"Contrast enhanced using {method}")
        return enhanced
    
    def adjust_brightness(self, image, alpha=1.0, beta=0):
        """
        Adjust image brightness and contrast.
        
        Args:
            image: Input image
            alpha: Contrast control (1.0-3.0)
            beta: Brightness control (-100 to +100)
        
        Returns:
            Adjusted image
        """
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        logger.info(f"Adjusted brightness: alpha={alpha}, beta={beta}")
        return adjusted
    
    def adaptive_threshold(self, image):
        """Apply adaptive thresholding for better text clarity."""
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        logger.info("Applied adaptive thresholding")
        return binary
    
def detect_and_correct_skew(self, image, max_angle=10):
    """
    Detect and correct skew angle with safety bounds.
    
    Args:
        image: Grayscale image
        max_angle: Maximum angle to correct (avoid extreme rotations)
        
    Returns:
        Deskewed image
    """
    # Ensure image is grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Threshold the image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Find all white pixels
    coords = np.column_stack(np.where(thresh > 0))
    
    if len(coords) < 5:
        logger.info("Not enough points to detect skew")
        return image
    
    # Calculate the angle
    angle = cv2.minAreaRect(coords)[-1]
    
    # Normalize angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Safety check: don't correct extreme angles
    if abs(angle) > max_angle:
        logger.warning(f"Skew angle {angle:.2f}° exceeds max {max_angle}°, skipping correction")
        return image
    
    # Only correct if significant (> 0.5 degrees)
    if abs(angle) > 0.5:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Calculate new image size to prevent cropping
        # This is the key fix!
        angle_rad = np.deg2rad(abs(angle))
        new_w = int(w * np.cos(angle_rad) + h * np.sin(angle_rad))
        new_h = int(w * np.sin(angle_rad) + h * np.cos(angle_rad))
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Adjust translation to center the rotated image
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        
        # Rotate with new dimensions (prevents cropping)
        rotated = cv2.warpAffine(
            image, M, (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255  # White background
        )
        
        logger.info(f"Corrected skew: {angle:.2f}° (expanded canvas to prevent crop)")
        return rotated
    else:
        logger.info("No significant skew detected")
        return image
    
    def resize_image(self, image, target_width=None, target_height=None):
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: Input image
            target_width: Target width
            target_height: Target height
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        
        if target_width and not target_height:
            ratio = target_width / w
            target_height = int(h * ratio)
        elif target_height and not target_width:
            ratio = target_height / h
            target_width = int(w * ratio)
        elif not target_width and not target_height:
            return image
        
        resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        logger.info(f"Resized image: {w}x{h} -> {target_width}x{target_height}")
        return resized
    
    def sharpen_image(self, image):
        """
        Sharpen image to enhance text clarity.
        
        Args:
            image: Grayscale image
            
        Returns:
            Sharpened image
        """
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        logger.info("Image sharpened")
        return sharpened
    
    def remove_borders(self, image, border_size=10):
        """
        Remove borders from image.
        
        Args:
            image: Input image
            border_size: Pixels to remove from each side
            
        Returns:
            Cropped image
        """
        h, w = image.shape[:2]
        cropped = image[border_size:h-border_size, border_size:w-border_size]
        logger.info(f"Removed {border_size}px borders")
        return cropped
    
    def preprocess_for_ocr(self, image, apply_threshold=False):
        """
        Full preprocessing pipeline optimized for EasyOCR.
        
        Args:
            image: Input image (PIL or numpy)
            apply_threshold: Whether to apply binarization
            
        Returns:
            Preprocessed image (numpy array)
        """
        logger.info("Starting preprocessing pipeline")
        
        # Convert PIL to OpenCV if needed
        if isinstance(image, Image.Image):
            image = self.pil_to_cv2(image)
        
        # Convert to grayscale
        gray = self.convert_to_grayscale(image)
        
        # Remove noise
        denoised = self.remove_noise(gray)
        
        # Enhance contrast
        enhanced = self.enhance_contrast(denoised, method='clahe')
        
        # Correct skew
        deskewed = self.detect_and_correct_skew(enhanced)
        
        # Optional: Apply thresholding for very clear text
        if apply_threshold:
            deskewed = self.adaptive_threshold(deskewed)
        
        logger.info("Preprocessing complete")
        return deskewed
