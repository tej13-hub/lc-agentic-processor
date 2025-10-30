"""Test grayscale to RGB conversion"""

import cv2
import numpy as np
from paddleocr import PaddleOCR

print("Testing grayscale conversion for PaddleOCR\n")

# Load your preprocessed image
gray = cv2.imread('debug_preprocessed.png', cv2.IMREAD_GRAYSCALE)

print(f"Loaded grayscale image: {gray.shape}, dtype: {gray.dtype}")
print(f"Mean: {gray.mean():.1f}, Std: {gray.std():.1f}\n")

# Initialize OCR
ocr = PaddleOCR(lang='en')

# Method 1: Stack (old way)
print("Method 1: np.stack")
rgb_stack = np.stack([gray] * 3, axis=-1)
print(f"  Shape: {rgb_stack.shape}")
result1 = ocr.ocr(rgb_stack)
lines1 = len(result1[0]) if result1 and result1[0] else 0
print(f"  Result: {lines1} lines detected\n")

# Method 2: OpenCV conversion (correct way)
print("Method 2: cv2.cvtColor")
rgb_cvt = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
print(f"  Shape: {rgb_cvt.shape}")
result2 = ocr.ocr(rgb_cvt)
lines2 = len(result2[0]) if result2 and result2[0] else 0
print(f"  Result: {lines2} lines detected\n")

# Method 3: OpenCV conversion with BGR intermediate
print("Method 3: cv2.cvtColor via BGR")
bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
rgb_bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
print(f"  Shape: {rgb_bgr.shape}")
result3 = ocr.ocr(rgb_bgr)
lines3 = len(result3[0]) if result3 and result3[0] else 0
print(f"  Result: {lines3} lines detected\n")

# Compare
print("="*50)
print("COMPARISON:")
print(f"  Method 1 (stack):        {lines1} lines")
print(f"  Method 2 (GRAY2RGB):     {lines2} lines")
print(f"  Method 3 (via BGR):      {lines3} lines")
print("="*50)

# Show sample text from best method
best_result = max([(result1, lines1, "Method 1"), 
                   (result2, lines2, "Method 2"), 
                   (result3, lines3, "Method 3")], 
                  key=lambda x: x[1])

if best_result[1] > 0:
    print(f"\nBest: {best_result[2]}")
    print("Sample text:")
    for i, line in enumerate(best_result[0][0][:5]):
        print(f"  {i+1}. {line[1][0]}")
else:
    print("\n✗ None of the methods worked!")
    print("Possible issues:")
    print("  - Image quality too poor")
    print("  - Text too small")
    print("  - Wrong language")
