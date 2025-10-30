"""
Preprocessing Agent - Makes intelligent preprocessing decisions using LLM
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any
from PIL import Image

from src.llm.llama_client import LlamaClient
from src.preprocessing.image_preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)


class PreprocessingAgent:
    """
    Intelligent preprocessing agent with conservative approach.
    """
    
    def __init__(self, llm_client: LlamaClient):
        self.llm = llm_client
        self.preprocessor = ImagePreprocessor()
        self.name = "PreprocessingAgent"
        logger.info(f"✓ {self.name} initialized")
    
    def analyze_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image quality metrics.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            dict: Quality metrics
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate quality metrics
        metrics = {
            "brightness": float(np.mean(gray)),
            "contrast": float(np.std(gray)),
            "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "noise_level": self._estimate_noise(gray),
            "skew_angle": self._estimate_skew_angle(gray),
            "resolution": gray.shape
        }
        
        return metrics
    
    def _estimate_noise(self, image: np.ndarray) -> float:
        """Estimate noise level in image."""
        kernel = np.ones((5, 5), np.float32) / 25
        smoothed = cv2.filter2D(image, -1, kernel)
        noise = np.std(image - smoothed)
        return float(noise)
    
    def _estimate_skew_angle(self, image: np.ndarray) -> float:
        """Estimate skew angle (return actual angle, not just boolean)."""
        # Threshold the image
        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Find all white pixels
        coords = np.column_stack(np.where(thresh > 0))
        
        if len(coords) < 5:
            return 0.0
        
        # Calculate angle
        angle = cv2.minAreaRect(coords)[-1]
        
        # Normalize angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        return float(angle)
    
    def decide_preprocessing_strategy(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use simple rule-based strategy (skip LLM to avoid errors initially).
        
        Args:
            metrics: Image quality metrics
            
        Returns:
            dict: Preprocessing decisions
        """
        logger.info(f"{self.name}: Analyzing image and deciding strategy...")
        
        # Conservative rule-based decisions
        decisions = {
            "needs_brightness_adjustment": False,  # Usually not needed
            "brightness_adjustment_level": 0,
            "needs_denoising": metrics['noise_level'] > 12,  # Only if very noisy
            "denoising_strength": "low" if metrics['noise_level'] > 12 else "none",
            "needs_contrast_enhancement": metrics['contrast'] < 35,  # Only if low contrast
            "contrast_method": "clahe",
            "needs_skew_correction": abs(metrics['skew_angle']) > 2.0,  # Only if > 2 degrees
            "needs_thresholding": False,  # Usually degrades OCR quality
            "threshold_method": "none",
            "reasoning": f"Conservative preprocessing: noise={metrics['noise_level']:.1f}, contrast={metrics['contrast']:.1f}, skew={metrics['skew_angle']:.2f}°"
        }
        
        logger.info(f"  Reasoning: {decisions['reasoning']}")
        return decisions
    
    def preprocess(self, image, decisions: Dict[str, Any]) -> np.ndarray:
        """
        Execute preprocessing based on agent's decisions.
        
        Args:
            image: Input image (PIL or numpy)
            decisions: Preprocessing decisions
            
        Returns:
            Preprocessed image (numpy array)
        """
        logger.info(f"{self.name}: Executing minimal preprocessing...")
        
        # Convert PIL to OpenCV if needed
        if isinstance(image, Image.Image):
            image = self.preprocessor.pil_to_cv2(image)
        
        # Convert to grayscale (this is essential)
        processed = self.preprocessor.convert_to_grayscale(image)
        
        # Only apply denoising if really needed
        if decisions.get('needs_denoising') and decisions.get('denoising_strength') != 'none':
            strength_map = {"low": 5, "medium": 10, "high": 15}
            strength = strength_map.get(decisions.get('denoising_strength', 'low'), 5)
            processed = cv2.fastNlMeansDenoising(processed, h=strength)
            logger.info(f"  ✓ Denoised: strength {strength}")
        
        # Only enhance contrast if needed
        if decisions.get('needs_contrast_enhancement'):
            processed = self.preprocessor.enhance_contrast(processed, method='clahe')
            logger.info(f"  ✓ Enhanced contrast")
        
        # SKIP skew correction for now (it's causing cropping issues)
        # if decisions.get('needs_skew_correction'):
        #     processed = self.preprocessor.detect_and_correct_skew(processed)
        #     logger.info(f"  ✓ Corrected skew")
        
        # NEVER apply thresholding (degrades OCR)
        
        logger.info(f"✓ Minimal preprocessing complete")
        return processed
    
    def process(self, image) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        Full preprocessing agent workflow: analyze → decide → execute.
        
        Args:
            image: Input image
            
        Returns:
            tuple: (preprocessed_image, decisions)
        """
        logger.info(f"\n{self.name}: Starting minimal preprocessing...")
        
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            img_array = self.preprocessor.pil_to_cv2(image)
        else:
            img_array = image
        
        # Step 1: Analyze image quality
        metrics = self.analyze_image_quality(img_array)
        logger.info(f"  Image metrics:")
        logger.info(f"    Brightness: {metrics['brightness']:.1f}")
        logger.info(f"    Contrast: {metrics['contrast']:.1f}")
        logger.info(f"    Sharpness: {metrics['sharpness']:.1f}")
        logger.info(f"    Noise: {metrics['noise_level']:.2f}")
        logger.info(f"    Skew: {metrics['skew_angle']:.2f}°")
        
        # Step 2: Decide strategy (rule-based, not LLM for now)
        decisions = self.decide_preprocessing_strategy(metrics)
        
        # Step 3: Execute minimal preprocessing
        preprocessed = self.preprocess(image, decisions)
        
        # Add metrics to decisions
        decisions['original_metrics'] = metrics
        
        return preprocessed, decisions
