"""
Test script to verify all dependencies are working
"""

import sys
import os

def test_imports():
    """Test all critical imports"""
    
    print("="*70)
    print("TESTING DEPENDENCIES")
    print("="*70)
    
    tests = [
        ('requests', 'HTTP client'),
        ('dotenv', 'Environment variables'),
        ('pydantic', 'Data validation'),
        ('PIL', 'Image processing (Pillow)'),
        ('cv2', 'Computer vision (OpenCV)'),
        ('pdf2image', 'PDF conversion'),
        ('paddleocr', 'OCR engine'),
        ('yaml', 'YAML parsing'),
        ('numpy', 'Numerical computing'),
    ]
    
    failed = []
    
    for module, description in tests:
        try:
            __import__(module)
            print(f"✓ {description:<30} ({module})")
        except ImportError as e:
            print(f"✗ {description:<30} ({module}) - FAILED")
            failed.append((module, str(e)))
    
    print("\n" + "="*70)
    
    if failed:
        print("❌ FAILED IMPORTS:")
        for module, error in failed:
            print(f"  - {module}: {error}")
        sys.exit(1)
    else:
        print("✅ ALL DEPENDENCIES INSTALLED SUCCESSFULLY!")
    
    # Test Python version
    print(f"\n✓ Python version: {sys.version}")
    
    # Test Ollama connection
    print("\nTesting Ollama connection...")
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            print("✓ Ollama is running")
            models = response.json().get('models', [])
            if models:
                print(f"✓ Available models: {', '.join([m['name'] for m in models])}")
            else:
                print("⚠ No models found. Run: ollama pull llama3.2:3b")
        else:
            print("✗ Ollama returned error")
    except Exception as e:
        print(f"✗ Cannot connect to Ollama: {e}")
        print("  Make sure Ollama is running: https://ollama.com")

if __name__ == '__main__':
    test_imports()
