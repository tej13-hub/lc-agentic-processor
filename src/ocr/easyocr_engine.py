"""
EasyOCR engine for text extraction from documents
Supports typed and handwritten text
"""

import easyocr
import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class EasyOCREngine:
    """
    EasyOCR-based text extraction engine.
    Excellent for multilingual and handwritten text.
    """
    
    def __init__(self, languages=['en'], gpu=False):
        """
        Initialize EasyOCR reader.
        
        Args:
            languages (list): List of language codes (e.g., ['en', 'hi', 'ar'])
            gpu (bool): Use GPU acceleration if available
        """
        self.languages = languages
        self.gpu = gpu
        
        logger.info(f"Initializing EasyOCR with languages: {languages}")
        logger.info(f"GPU acceleration: {gpu}")
        
        try:
            self.reader = easyocr.Reader(
                languages,
                gpu=gpu,
                verbose=False
            )
            logger.info("✓ EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"EasyOCR initialization failed: {e}")
            raise
    
    def extract_text(self, image: np.ndarray, detail=0) -> str:
        """
        Extract all text from image.
        
        Args:
            image: Input image (numpy array)
            detail: 0=text only, 1=text+confidence, 2=text+bbox+confidence
            
        Returns:
            str: Extracted text
        """
        try:
            logger.info("Extracting text with EasyOCR...")
            
            results = self.reader.readtext(image, detail=1)
            
            # Extract text only
            text_lines = [result[1] for result in results]
            text = "\n".join(text_lines)
            
            logger.info(f"Extracted {len(text_lines)} lines, {len(text)} characters")
            return text.strip()
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return ""
    
    def extract_structured(self, image: np.ndarray) -> Dict:
        """
        Extract text with bounding boxes and confidence scores.
        
        Args:
            image: Input image
            
        Returns:
            dict: Structured OCR results
        """
        try:
            logger.info("Extracting structured text with EasyOCR...")
            
            results = self.reader.readtext(image, detail=1)
            
            structured_data = {
                "text": [],
                "boxes": [],
                "confidences": [],
                "full_text": ""
            }
            
            for bbox, text, confidence in results:
                structured_data['text'].append(text)
                structured_data['boxes'].append(bbox)
                structured_data['confidences'].append(float(confidence))
            
            # Combine all text
            structured_data['full_text'] = "\n".join(structured_data['text'])
            
            # Calculate average confidence
            if structured_data['confidences']:
                avg_conf = sum(structured_data['confidences']) / len(structured_data['confidences'])
                structured_data['average_confidence'] = avg_conf
                logger.info(f"Average OCR confidence: {avg_conf:.2%}")
            
            return structured_data
            
        except Exception as e:
            logger.error(f"Structured extraction failed: {e}")
            return {
                "text": [],
                "boxes": [],
                "confidences": [],
                "full_text": "",
                "error": str(e)
            }
    
    def extract_with_language_detection(self, image: np.ndarray) -> Tuple[str, str]:
        """
        Extract text and detect primary language.
        
        Args:
            image: Input image
            
        Returns:
            tuple: (extracted_text, detected_language)
        """
        results = self.reader.readtext(image, detail=1)
        
        # For simplicity, return the configured language
        # EasyOCR doesn't provide explicit language detection per text
        text = "\n".join([result[1] for result in results])
        primary_lang = self.languages[0] if self.languages else 'unknown'
        
        return text.strip(), primary_lang
