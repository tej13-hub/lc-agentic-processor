"""
PaddleOCR Engine - Compatible with PaddleX OCRResult dictionary format
With suppressed verbose logging
"""

import logging
from typing import Dict, Any
import numpy as np
import cv2
from src.ocr.base_ocr import BaseOCREngine

# Configure PaddleOCR logging to be less verbose
import os
os.environ['FLAGS_log_level'] = '3'  # Only show errors

# Suppress PaddleOCR's verbose output
logging.getLogger('ppocr').setLevel(logging.ERROR)
logging.getLogger('paddlex').setLevel(logging.ERROR)
logging.getLogger('paddle').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR engine compatible with PaddleX OCRResult format."""
    
    def __init__(self, languages=['en'], gpu=False, **kwargs):
        super().__init__(languages)
        
        logger.info("Initializing PaddleOCR...")
        
        try:
            # Suppress warnings during initialization
            import warnings
            warnings.filterwarnings('ignore')
            
            from paddleocr import PaddleOCR
            
            # Initialize without show_log parameter (doesn't exist)
            self.ocr = PaddleOCR(lang='en')
            
            logger.info("✓ PaddleOCR initialized successfully")
            
        except Exception as e:
            logger.error(f"PaddleOCR initialization failed: {e}")
            raise
    
    def _parse_paddlex_result(self, result):
        """
        Parse PaddleX OCRResult dictionary.
        
        Keys in OCRResult:
        - rec_texts: list of text strings
        - rec_scores: list of confidence scores
        - rec_boxes: bounding boxes
        """
        try:
            # PaddleX OCRResult is dict-like with keys()
            if hasattr(result, 'keys'):
                logger.debug("  Detected PaddleX OCRResult dictionary format")
                
                # Access via dictionary keys
                texts = result.get('rec_texts', [])
                scores = result.get('rec_scores', [])
                boxes = result.get('rec_boxes', [])
                
                logger.debug(f"  Extracted: {len(texts)} texts, {len(scores)} scores")
                
                return texts, scores, boxes
            
            # Standard PaddleOCR format: [[bbox, (text, conf)], ...]
            elif isinstance(result, list) and len(result) > 0:
                logger.debug("  Detected standard PaddleOCR list format")
                
                texts = [line[1][0] for line in result if line and len(line) >= 2]
                scores = [line[1][1] for line in result if line and len(line) >= 2]
                boxes = [line[0] for line in result if line and len(line) >= 2]
                
                return texts, scores, boxes
            
            else:
                logger.warning(f"  Unknown result format: {type(result)}")
                return [], [], []
                
        except Exception as e:
            logger.error(f"  Error parsing result: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [], [], []
    
    def extract_text(self, image: np.ndarray) -> str:
        """Extract text from image."""
        try:
            logger.info("Extracting text with PaddleOCR...")
            logger.debug(f"  Input: shape={image.shape}, dtype={image.dtype}")
            
            # Prepare image - convert grayscale to BGR
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                gray = np.squeeze(image, axis=2)
                image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # Ensure uint8
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            logger.debug(f"  Prepared: shape={image.shape}, dtype={image.dtype}")
            
            # Run OCR
            result = self.ocr.ocr(image)
            
            logger.debug(f"  Result type: {type(result)}")
            
            # Handle empty results
            if result is None or (isinstance(result, list) and len(result) == 0):
                logger.warning("  No text detected (empty result)")
                return ""
            
            # Get first element if result is list
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            
            # Parse result
            texts, scores, boxes = self._parse_paddlex_result(result)
            
            if not texts:
                logger.warning("  No text extracted after parsing")
                return ""
            
            full_text = "\n".join(texts)
            logger.info(f"✓ Extracted {len(texts)} lines, {len(full_text)} characters")
            
            # Show sample (only first line for brevity)
            if texts:
                conf = scores[0] if scores else 0
                logger.debug(f"    Sample: [{conf:.2%}] {texts[0][:50]}...")
            
            return full_text.strip()
            
        except Exception as e:
            logger.error(f"✗ Text extraction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""
    
    def extract_structured(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract structured data."""
        try:
            logger.info("Extracting structured text with PaddleOCR...")
            logger.debug(f"  Input: shape={image.shape}, dtype={image.dtype}")
            
            # Prepare image
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                gray = np.squeeze(image, axis=2)
                image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Run OCR
            result = self.ocr.ocr(image)
            
            structured_data = {
                "text": [],
                "boxes": [],
                "confidences": [],
                "full_text": "",
                "average_confidence": 0.0
            }
            
            if result is None or (isinstance(result, list) and len(result) == 0):
                logger.warning("  No text detected")
                return structured_data
            
            # Get first element if list
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            
            # Parse result
            texts, scores, boxes = self._parse_paddlex_result(result)
            
            if not texts:
                logger.warning("  No text extracted")
                return structured_data
            
            logger.info(f"  Parsed {len(texts)} text regions")
            
            # Fill structured data
            structured_data['text'] = texts
            structured_data['boxes'] = boxes if isinstance(boxes, list) else boxes.tolist()
            structured_data['confidences'] = [float(s) for s in scores]
            structured_data['full_text'] = "\n".join(texts)
            
            # Calculate average confidence
            if structured_data['confidences']:
                avg_conf = sum(structured_data['confidences']) / len(structured_data['confidences'])
                structured_data['average_confidence'] = avg_conf
                logger.info(f"✓ Extracted {len(texts)} lines")
                logger.info(f"✓ Average confidence: {avg_conf:.2%}")
                
                # Show samples (only first 3 for brevity)
                for i in range(min(3, len(texts))):
                    conf = structured_data['confidences'][i]
                    text = texts[i][:50]
                    logger.debug(f"    {i+1}. [{conf:.2%}] {text}")
            else:
                logger.warning("  No confidence scores")
            
            return structured_data
            
        except Exception as e:
            logger.error(f"✗ Structured extraction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "text": [], "boxes": [], "confidences": [],
                "full_text": "", "average_confidence": 0.0,
                "error": str(e)
            }
