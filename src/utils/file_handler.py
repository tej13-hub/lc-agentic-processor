"""
File handling utilities for document processing
"""

import os
import sys
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileHandler:
    """Handles file I/O operations for document processing."""
    
    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
    SUPPORTED_PDF_FORMAT = '.pdf'
    
    # Poppler path for Windows (modify this if needed)
    POPPLER_PATH = r"C:\Program Files\Release-25.07.0-0\poppler-25.07.0\Library\bin"
    
    # Auto-detect poppler on Windows
    if sys.platform == 'win32':
        possible_paths = [
            r'C:\Program Files\poppler\Library\bin',
            r'C:\Program Files (x86)\poppler\Library\bin',
            r'C:\poppler\Library\bin',
        ]
        for path in possible_paths:
            if os.path.exists(path):
                POPPLER_PATH = path
                logger.info(f"Found poppler at: {path}")
                break
    
    @staticmethod
    def validate_file(file_path):
        """
        Validate file existence and format.
        
        Args:
            file_path (str): Path to file
            
        Returns:
            tuple: (is_valid, file_type)
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False, None
        
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext in FileHandler.SUPPORTED_IMAGE_FORMATS:
            return True, 'image'
        elif file_ext == FileHandler.SUPPORTED_PDF_FORMAT:
            return True, 'pdf'
        else:
            logger.error(f"Unsupported format: {file_ext}")
            return False, None
    
    @staticmethod
    def pdf_to_images(pdf_path, dpi=300):
        """
        Convert PDF to images.
        
        Args:
            pdf_path (str): Path to PDF
            dpi (int): Resolution
            
        Returns:
            list: PIL Images
        """
        try:
            logger.info(f"Converting PDF to images: {pdf_path}")
            
            # Convert with or without poppler_path
            if FileHandler.POPPLER_PATH:
                images = convert_from_path(
                    pdf_path, 
                    dpi=dpi,
                    poppler_path=FileHandler.POPPLER_PATH
                )
            else:
                images = convert_from_path(pdf_path, dpi=dpi)
            
            logger.info(f"Converted {len(images)} page(s)")
            return images
            
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            logger.error("Make sure poppler is installed:")
            logger.error("  Ubuntu/Debian: sudo apt-get install poppler-utils")
            logger.error("  macOS: brew install poppler")
            logger.error("  Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/")
            return []
    
    @staticmethod
    def load_image(image_path):
        """Load image from file."""
        try:
            img = Image.open(image_path)
            logger.info(f"Loaded image: {image_path}, Size: {img.size}")
            return img
        except Exception as e:
            logger.error(f"Image load failed: {e}")
            return None
    
    @staticmethod
    def save_image(image, output_path):
        """Save image to file."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if isinstance(image, Image.Image):
                image.save(output_path)
            else:
                # numpy array
                Image.fromarray(image).save(output_path)
            
            logger.info(f"Saved: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Save failed: {e}")
            return False
