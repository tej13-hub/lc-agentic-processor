"""
Test MCP Schema Fetching
"""

import httpx
import json
from config.settings import settings


def test_fetch_schema():
    """Test fetching schema from MCP server."""
    
    print("="*70)
    print("TESTING MCP SCHEMA FETCH")
    print("="*70)
    
    mcp_url = settings.MCP_SERVER_URL
    endpoint = f"{mcp_url}/tools/list"
    
    print(f"\nMCP Server: {mcp_url}")
    print(f"Endpoint: {endpoint}")
    
    try:
        # Fetch tools list
        print("\n1. Fetching tools list...")
        response = httpx.get(endpoint, timeout=10.0)
        response.raise_for_status()
        
        data = response.json()
        tools = data.get('tools', [])
        
        print(f"   ✓ Response: {response.status_code}")
        print(f"   ✓ Found {len(tools)} tool(s)")
        
        # Print each tool
        print("\n2. Available tools:")
        for i, tool in enumerate(tools, 1):
            tool_name = tool.get('name', 'unknown')
            print(f"\n   Tool {i}: {tool_name}")
            
            # Print schema info
            schema = tool.get('schema', {})
            if schema:
                required = schema.get('required_fields', [])
                optional = schema.get('optional_fields', [])
                print(f"      Required fields: {required}")
                print(f"      Optional fields: {optional}")
            
            # Print sample payload
            sample = tool.get('sample_payload', {})
            if sample:
                print(f"      Sample payload: {len(sample)} fields")
        
        # Try to get specific tool
        print("\n3. Testing tool lookup:")
        tool_name = "setDocumentDetails"
        
        found = None
        for tool in tools:
            if tool.get('name') == tool_name:
                found = tool
                break
        
        if found:
            print(f"   ✓ Found '{tool_name}'")
            print(f"\n   Full config:")
            print(json.dumps(found, indent=2))
        else:
            print(f"   ✗ Tool '{tool_name}' not found")
        
        print("\n" + "="*70)
        print("✓ TEST PASSED")
        print("="*70)
        
        return True
    
    except httpx.TimeoutException:
        print(f"\n✗ FAILED: Timeout connecting to {endpoint}")
        print("   Is the MCP server running?")
        return False
    
    except httpx.HTTPStatusError as e:
        print(f"\n✗ FAILED: HTTP {e.response.status_code}")
        print(f"   Response: {e.response.text}")
        return False
    
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_fetch_schema()
    exit(0 if success else 1)
