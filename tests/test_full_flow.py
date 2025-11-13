"""
Test complete flow: Fetch schema → Build payload → Validate
"""

import sys
sys.path.append('.')

from src.payload.payload_builder import PayloadBuilder
from src.llm.llm_factory import create_llm_client
from config.settings import settings


def test_full_flow():
    """Test the complete payload building flow."""
    
    print("="*70)
    print("TESTING COMPLETE PAYLOAD FLOW")
    print("="*70)
    
    # Initialize
    llm_client = create_llm_client()
    payload_builder = PayloadBuilder(llm_client)
    
    # Sample extracted fields
    extracted_fields = {
        'invoice_number': 'INV-2024-001',
        'amount': 50000.00,
        'date': '2024-11-13',
        'exporter': 'ABC Corp',
        'importer': 'XYZ Ltd',
        'lc_number': 'LC-2024-12345'
    }
    
    print("\nExtracted fields:")
    for key, value in extracted_fields.items():
        print(f"  {key}: {value}")
    
    # Build payload
    print("\nBuilding payload...")
    result = payload_builder.build_payload(
        tool_name='setDocumentDetails',
        extracted_fields=extracted_fields
    )
    
    if result['success']:
        print("\n✓ Payload built successfully!")
        
        payload = result['payload']
        validation = result['validation']
        
        print(f"\nPayload ({len(payload)} fields):")
        for key, value in payload.items():
            source = "📄 doc" if key in result['fields_from_doc'] else "📋 sample"
            print(f"  {source} {key}: {value}")
        
        print(f"\nValidation: {'✓ PASSED' if validation['valid'] else '✗ FAILED'}")
        
        if validation['errors']:
            print(f"\nErrors:")
            for error in validation['errors']:
                print(f"  ✗ {error}")
        
        if validation['warnings']:
            print(f"\nWarnings:")
            for warning in validation['warnings']:
                print(f"  ⚠ {warning}")
        
        print("\n" + "="*70)
        print("✓ TEST PASSED")
        print("="*70)
        
        return True
    
    else:
        print(f"\n✗ FAILED: {result['error']}")
        print(f"   Message: {result.get('message')}")
        return False


if __name__ == '__main__':
    success = test_full_flow()
    exit(0 if success else 1)
