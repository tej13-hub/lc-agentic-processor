"""
OCR Factory - Creates OCR engine based on configuration
"""

import logging
from src.ocr.base_ocr import BaseOCREngine
from config.settings import settings

logger = logging.getLogger(__name__)


def create_ocr_engine() -> BaseOCREngine:
    """
    Factory function to create OCR engine based on settings.
    
    Returns:
        BaseOCREngine: Configured OCR engine
    """
    engine_name = settings.OCR_ENGINE.lower()
    languages = settings.ocr_language_list
    gpu = settings.OCR_GPU
    
    logger.info(f"Creating OCR engine: {engine_name}")
    
    if engine_name == 'easyocr':
        from src.ocr.easyocr_engine import EasyOCREngine
        return EasyOCREngine(languages=languages, gpu=gpu)
    
    elif engine_name == 'paddleocr':
        from src.ocr.paddleocr_engine import PaddleOCREngine
        return PaddleOCREngine(languages=languages, gpu=gpu)
    
    else:
        logger.warning(f"Unknown OCR engine: {engine_name}, defaulting to EasyOCR")
        from src.ocr.easyocr_engine import EasyOCREngine
        return EasyOCREngine(languages=languages, gpu=gpu)
