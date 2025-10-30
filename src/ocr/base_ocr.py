"""
Base OCR interface for plug-and-play OCR engines
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class BaseOCREngine(ABC):
    """Abstract base class for OCR engines."""
    
    def __init__(self, languages=None, **kwargs):
        self.languages = languages or ['en']
    
    @abstractmethod
    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract text from image.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            str: Extracted text
        """
        pass
    
    @abstractmethod
    def extract_structured(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract text with bounding boxes and confidence.
        
        Args:
            image: Input image
            
        Returns:
            dict: Structured OCR results
        """
        pass
