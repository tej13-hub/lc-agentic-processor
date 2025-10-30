"""
Test LLM configuration - Local and Remote
"""

import os
import sys

def test_local_llm():
    """Test local LLM configuration."""
    print("\n" + "="*70)
    print("TESTING LOCAL LLM (Ollama)")
    print("="*70)
    
    # Set environment
    os.environ['LLM_TYPE'] = 'local'
    
    # Import after setting env
    from src.llm.llm_factory import create_llm_client
    
    try:
        llm = create_llm_client()
        print(f"✓ LLM Client created: {type(llm).__name__}")
        
        # Test generation
        response = llm.generate("Say 'Hello from local LLM!'", timeout=30)
        print(f"✓ Response: {response[:100]}...")
        
        # Test JSON generation
        json_response = llm.generate_json(
            '{"test": "value"}. Return this as JSON.',
            system_prompt="Return only valid JSON",
            timeout=30
        )
        print(f"✓ JSON Response: {json_response}")
        
        return True
        
    except Exception as e:
        print(f"✗ Local LLM test failed: {e}")
        return False


def test_remote_llm():
    """Test remote LLM configuration."""
    print("\n" + "="*70)
    print("TESTING REMOTE LLM (API)")
    print("="*70)
    
    # Set environment
    os.environ['LLM_TYPE'] = 'remote'
    os.environ['REMOTE_LLM_API_KEY'] = 'test_key'  # Set your real key
    
    # Import after setting env
    from src.llm.llm_factory import create_llm_client
    
    try:
        llm = create_llm_client()
        print(f"✓ LLM Client created: {type(llm).__name__}")
        
        print("⚠ Skipping API call test (set real API key to test)")
        print("  To test: Set REMOTE_LLM_API_KEY in .env and uncomment test code")
        
        # Uncomment to test with real API key:
        # response = llm.generate("Say 'Hello from remote LLM!'", timeout=30)
        # print(f"✓ Response: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Remote LLM test failed: {e}")
        return False


if __name__ == '__main__':
    print("="*70)
    print("LLM CONFIGURATION TEST")
    print("="*70)
    
    # Test local
    local_ok = test_local_llm()
    
    # Test remote
    remote_ok = test_remote_llm()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Local LLM:  {'✓ PASS' if local_ok else '✗ FAIL'}")
    print(f"Remote LLM: {'✓ PASS' if remote_ok else '✗ FAIL'}")
    print("="*70)
