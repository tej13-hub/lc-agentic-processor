"""Diagnose OCR issues"""

import sys
import cv2
import numpy as np
from PIL import Image

# Add to path
sys.path.insert(0, '.')

from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.ocr.ocr_factory import create_ocr_engine
from config.settings import settings

def diagnose(image_path):
    """Diagnose OCR extraction."""
    
    print("\n" + "="*70)
    print("OCR DIAGNOSTIC TOOL")
    print("="*70)
    
    # Load image
    print(f"\n1. Loading image: {image_path}")
    
    # Check if PDF or image
    from src.utils.file_handler import FileHandler
    file_handler = FileHandler()
    
    is_valid, file_type = file_handler.validate_file(image_path)
    if not is_valid:
        print(f"   ✗ Invalid file")
        return
    
    if file_type == 'pdf':
        print(f"   File type: PDF")
        pages = file_handler.pdf_to_images(image_path)
        if not pages:
            print(f"   ✗ Failed to load PDF")
            return
        img = pages[0]
        print(f"   Using first page")
    else:
        print(f"   File type: Image")
        img = file_handler.load_image(image_path)
    
    print(f"   Size: {img.size}")
    print(f"   Mode: {img.mode}")
    
    # Preprocess
    print(f"\n2. Preprocessing...")
    preprocessor = ImagePreprocessor()
    
    # Convert PIL to numpy
    img_array = preprocessor.pil_to_cv2(img)
    print(f"   Original shape: {img_array.shape}")
    
    # Method 1: Basic preprocessing (if available)
    try:
        preprocessed = preprocessor.preprocess_basic(img, return_pil=False)
        print(f"   ✓ Used preprocess_basic()")
    except AttributeError:
        # Fallback: Manual preprocessing
        print(f"   Using manual preprocessing...")
        preprocessed = preprocessor.convert_to_grayscale(img_array)
        preprocessed = preprocessor.remove_noise(preprocessed)
        preprocessed = preprocessor.enhance_contrast(preprocessed)
        print(f"   ✓ Manual preprocessing complete")
    
    # Save preprocessed for inspection
    cv2.imwrite('debug_preprocessed.png', preprocessed)
    print(f"   Saved: debug_preprocessed.png")
    print(f"   Shape: {preprocessed.shape}")
    print(f"   Mean brightness: {np.mean(preprocessed):.1f}")
    print(f"   Std dev: {np.std(preprocessed):.1f}")
    
    # Check image quality
    if np.mean(preprocessed) < 30:
        print(f"   ⚠ WARNING: Image very dark (mean brightness: {np.mean(preprocessed):.1f})")
    if np.std(preprocessed) < 10:
        print(f"   ⚠ WARNING: Very low contrast (std dev: {np.std(preprocessed):.1f})")
    
    # OCR
    print(f"\n3. Running OCR with {settings.OCR_ENGINE}...")
    ocr_engine = create_ocr_engine()
    
    # Test with preprocessed image
    print(f"   Extracting structured data...")
    result = ocr_engine.extract_structured(preprocessed)
    
    print(f"\n4. OCR Results:")
    print(f"   Lines detected: {len(result['text'])}")
    print(f"   Total characters: {len(result['full_text'])}")
    print(f"   Average confidence: {result.get('average_confidence', 0):.2%}")
    
    if result['text']:
        print(f"\n5. First 10 lines:")
        for i, (text, conf) in enumerate(zip(result['text'][:10], result['confidences'][:10])):
            print(f"   {i+1}. [{conf:.2%}] {text}")
        
        if len(result['text']) > 10:
            print(f"   ... and {len(result['text']) - 10} more lines")
    else:
        print(f"\n5. ✗ NO TEXT EXTRACTED!")
        print(f"\n   Possible reasons:")
        print(f"   • Image quality too poor")
        print(f"   • Text is too small/blurry")
        print(f"   • Wrong preprocessing (too dark/bright)")
        print(f"   • OCR engine issue")
        print(f"\n   Suggestions:")
        print(f"   • Check debug_preprocessed.png - is text visible?")
        print(f"   • Try higher DPI: TARGET_DPI=600 in .env")
        print(f"   • Try different OCR: OCR_ENGINE=tesseract")
    
    # Save full text
    with open('debug_ocr_output.txt', 'w', encoding='utf-8') as f:
        f.write(f"OCR Engine: {settings.OCR_ENGINE}\n")
        f.write(f"Confidence: {result.get('average_confidence', 0):.2%}\n")
        f.write(f"Lines: {len(result['text'])}\n")
        f.write(f"Characters: {len(result['full_text'])}\n")
        f.write(f"\n{'='*70}\n")
        f.write(f"EXTRACTED TEXT:\n")
        f.write(f"{'='*70}\n\n")
        f.write(result['full_text'])
    
    print(f"\n6. Full text saved to: debug_ocr_output.txt")
    
    # Additional diagnostics
    print(f"\n7. Diagnostics:")
    if 'error' in result:
        print(f"   ✗ Error occurred: {result['error']}")
    
    print(f"\n   File locations:")
    print(f"   • Preprocessed image: debug_preprocessed.png")
    print(f"   • OCR output text: debug_ocr_output.txt")
    print(f"\n   Next steps:")
    print(f"   1. Open debug_preprocessed.png - can YOU read the text?")
    print(f"   2. If yes but OCR failed: Try different OCR engine")
    print(f"   3. If no: Improve preprocessing (increase DPI, adjust contrast)")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        diagnose(sys.argv[1])
    else:
        print("Usage: python diagnose_ocr.py path/to/document.pdf")
        print("   or: python diagnose_ocr.py path/to/image.jpg")
