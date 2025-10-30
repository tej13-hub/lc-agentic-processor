"""Test PaddleOCR on debug images - Complete version"""

import cv2
from paddleocr import PaddleOCR

print("Testing PaddleOCR on debug images...\n")

ocr = PaddleOCR(lang='en')

test_files = [
    'debug_step1_gray.png',
    'debug_step2_denoised.png',
    'debug_step3_enhanced.png'
]

for filename in test_files:
    print(f"\n{'='*70}")
    print(f"Testing: {filename}")
    print('='*70)
    
    try:
        # Load image
        img = cv2.imread(filename)
        
        if img is None:
            print(f"✗ Could not load {filename}")
            continue
        
        print(f"Image loaded: shape={img.shape}, dtype={img.dtype}")
        
        # Convert grayscale to BGR if needed
        if len(img.shape) == 2:
            print("Converting grayscale to BGR...")
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            print("Converting single channel to BGR...")
            gray = img.squeeze(axis=2)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        print(f"After conversion: shape={img.shape}")
        
        # Run OCR
        print("Running OCR...")
        result = ocr.ocr(img)
        
        print(f"\nResult analysis:")
        print(f"  Type: {type(result)}")
        print(f"  Length: {len(result) if result else 0}")
        
        if result and len(result) > 0:
            ocr_result = result[0]
            
            print(f"  result[0] type: {type(ocr_result)}")
            print(f"  result[0] class name: {ocr_result.__class__.__name__}")
            
            # Check if it's a dictionary-like object
            if hasattr(ocr_result, 'keys'):
                print(f"\n✓ Dictionary-like OCRResult detected")
                print(f"\nKeys in OCRResult:")
                for key in ocr_result.keys():
                    print(f"  - {key}")
                
                # Attempt text extraction using correct keys
                print(f"\n{'='*70}")
                print("EXTRACTING TEXT...")
                print('='*70)
                
                texts = []
                scores = []
                boxes = []
                
                # Access via dictionary keys (PaddleX format)
                if 'rec_texts' in ocr_result:
                    texts = ocr_result['rec_texts']
                    print(f"✓ Found 'rec_texts': {len(texts)} items")
                
                if 'rec_scores' in ocr_result:
                    scores = ocr_result['rec_scores']
                    print(f"✓ Found 'rec_scores': {len(scores)} items")
                
                if 'rec_boxes' in ocr_result:
                    boxes = ocr_result['rec_boxes']
                    print(f"✓ Found 'rec_boxes': {len(boxes)} items")
                
                # Display extracted text
                if texts:
                    print(f"\n{'='*70}")
                    print(f"✓ SUCCESS! Extracted {len(texts)} lines")
                    print('='*70)
                    
                    print(f"\nFirst 15 lines:")
                    for i in range(min(15, len(texts))):
                        conf = scores[i] if i < len(scores) else 0.0
                        text_preview = texts[i][:60] if len(texts[i]) > 60 else texts[i]
                        print(f"  {i+1:2d}. [{conf:5.1%}] {text_preview}")
                    
                    if len(texts) > 15:
                        print(f"  ... and {len(texts) - 15} more lines")
                    
                    # Calculate average confidence
                    if scores:
                        avg_conf = sum(scores) / len(scores)
                        print(f"\n✓ Average confidence: {avg_conf:.2%}")
                    
                    # Save all text to file
                    output_file = f'extracted_{filename}.txt'
                    with open(output_file, 'w', encoding='utf-8') as f:
                        for i, text in enumerate(texts):
                            conf = scores[i] if i < len(scores) else 0.0
                            f.write(f"[{conf:.2%}] {text}\n")
                    
                    print(f"\n✓ Full text saved to: {output_file}")
                    
                else:
                    print(f"\n✗ No text extracted (texts list is empty)")
            
            elif isinstance(ocr_result, list):
                # Standard PaddleOCR format
                print(f"\n✓ Standard PaddleOCR list format")
                
                if len(ocr_result) > 0:
                    print(f"  Detected {len(ocr_result)} lines")
                    print("\nFirst 10 lines:")
                    
                    for i in range(min(10, len(ocr_result))):
                        line = ocr_result[i]
                        try:
                            text = line[1][0]
                            conf = line[1][1]
                            print(f"  {i+1}. [{conf:.2%}] {text}")
                        except Exception as e:
                            print(f"  {i+1}. Error parsing: {e}")
                else:
                    print(f"\n✗ Empty list")
            
            else:
                print(f"\n✗ Unknown result format: {type(ocr_result)}")
        
        else:
            print(f"\n✗ Empty or None result")
        
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
print("\nNEXT STEPS:")
print("1. If text was extracted successfully above:")
print("   → Your OCR is working!")
print("   → Run: python main.py")
print("\n2. If no text was extracted:")
print("   → Check the extracted_*.txt files")
print("   → Or switch to Tesseract: OCR_ENGINE=tesseract in .env")
print("="*70 + "\n")
