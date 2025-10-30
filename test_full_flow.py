"""Test the complete OCR flow to find the issue"""

import sys
sys.path.insert(0, '.')

import cv2
import numpy as np
from PIL import Image

from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.ocr.ocr_factory import create_ocr_engine

print("\n" + "="*70)
print("COMPLETE OCR FLOW TEST")
print("="*70)

# Step 1: Load original image
input_file = sys.argv[1] if len(sys.argv) > 1 else "input\insurance_certificate.png"

print(f"\n1. Loading: {input_file}")

from src.utils.file_handler import FileHandler
file_handler = FileHandler()

is_valid, file_type = file_handler.validate_file(input_file)

if file_type == 'pdf':
    pages = file_handler.pdf_to_images(input_file)
    img = pages[0] if pages else None
else:
    img = file_handler.load_image(input_file)

if img is None:
    print("✗ Failed to load image")
    sys.exit(1)

print(f"   ✓ Loaded: {img.size}, mode: {img.mode}")

# Step 2: Preprocess
print(f"\n2. Preprocessing...")
preprocessor = ImagePreprocessor()

# Convert to OpenCV
img_cv = preprocessor.pil_to_cv2(img)
print(f"   Original: shape={img_cv.shape}, dtype={img_cv.dtype}")

# Grayscale
gray = preprocessor.convert_to_grayscale(img_cv)
print(f"   Grayscale: shape={gray.shape}, mean={np.mean(gray):.1f}")

# Save for inspection
cv2.imwrite('debug_step1_gray.png', gray)
print(f"   Saved: debug_step1_gray.png")

# Denoise
denoised = preprocessor.remove_noise(gray)
print(f"   Denoised: mean={np.mean(denoised):.1f}")
cv2.imwrite('debug_step2_denoised.png', denoised)

# Enhance
enhanced = preprocessor.enhance_contrast(denoised)
print(f"   Enhanced: mean={np.mean(enhanced):.1f}")
cv2.imwrite('debug_step3_enhanced.png', enhanced)

# Step 3: Test OCR directly on each stage
print(f"\n3. Testing OCR on different preprocessing stages...")

ocr_engine = create_ocr_engine()

# Test on grayscale
print(f"\n   A. Testing on grayscale...")
result_gray = ocr_engine.extract_structured(gray)
print(f"      Lines: {len(result_gray['text'])}")
print(f"      Confidence: {result_gray.get('average_confidence', 0):.2%}")

# Test on denoised
print(f"\n   B. Testing on denoised...")
result_denoised = ocr_engine.extract_structured(denoised)
print(f"      Lines: {len(result_denoised['text'])}")
print(f"      Confidence: {result_denoised.get('average_confidence', 0):.2%}")

# Test on enhanced
print(f"\n   C. Testing on enhanced...")
result_enhanced = ocr_engine.extract_structured(enhanced)
print(f"      Lines: {len(result_enhanced['text'])}")
print(f"      Confidence: {result_enhanced.get('average_confidence', 0):.2%}")

# Step 4: Show best result
print(f"\n4. Best Result:")

results = [
    ('grayscale', result_gray),
    ('denoised', result_denoised),
    ('enhanced', result_enhanced)
]

best = max(results, key=lambda x: len(x[1]['full_text']))
best_name, best_result = best

print(f"   Best preprocessing: {best_name}")
print(f"   Lines: {len(best_result['text'])}")
print(f"   Characters: {len(best_result['full_text'])}")
print(f"   Confidence: {best_result.get('average_confidence', 0):.2%}")

if best_result['text']:
    print(f"\n   First 5 lines:")
    for i, line in enumerate(best_result['text'][:5]):
        print(f"   {i+1}. {line}")
    
    # Save
    with open('debug_best_ocr_output.txt', 'w', encoding='utf-8') as f:
        f.write(best_result['full_text'])
    print(f"\n   ✓ Saved to: debug_best_ocr_output.txt")
else:
    print(f"\n   ✗ NO TEXT EXTRACTED IN ANY STAGE!")
    print(f"\n   Check these files:")
    print(f"   • debug_step1_gray.png")
    print(f"   • debug_step2_denoised.png")
    print(f"   • debug_step3_enhanced.png")
    print(f"\n   Can you read text in any of these images?")

print("\n" + "="*70 + "\n")
