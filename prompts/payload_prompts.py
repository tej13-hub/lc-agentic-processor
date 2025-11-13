"""
Payload Building Prompts - With Schema Resolution Support
"""

import json
from typing import Dict, Any


PAYLOAD_MAPPING_PROMPT = """
You are a data mapping assistant. Your job is to populate an API payload using extracted document data.

TOOL: {tool_name}

PAYLOAD SCHEMA (ALL FIELDS MUST BE POPULATED):
{schema_description}

CRITICAL RULES:
1. You MUST populate ALL fields from the schema above (including nested objects and arrays)
2. For each field:
   a. First check if it exists in EXTRACTED DATA below
   b. If found in extracted data → USE IT
   c. If NOT found in extracted data → USE VALUE FROM SAMPLE PAYLOAD below
3. Maintain exact structure:
   - Keep nested objects as objects
   - Keep arrays as arrays
   - Match field names exactly
4. Data types:
   - Fields marked "number" must be numeric (not string)
   - Fields marked "string" must be strings
   - Date fields must be in YYYY-MM-DD format
5. Do NOT skip any field - every field must be present
6. Do NOT add extra fields not in the schema
7. Return ONLY valid JSON (no explanations, no markdown)

EXTRACTED FROM DOCUMENT (USE THIS FIRST - HIGHEST PRIORITY):
{extracted_fields}

SAMPLE PAYLOAD (USE FOR ANY MISSING FIELDS):
{sample_payload}

EXAMPLE: If "invoice_number" is in extracted data, use that value. If "currency" is not in extracted data, use the sample value.

YOUR RESPONSE (ONLY THE COMPLETE JSON PAYLOAD, NOTHING ELSE):
"""


def build_payload_prompt(
    tool_name: str,
    resolved_schema: Dict[str, Any],
    extracted_fields: Dict[str, Any],
    sample_payload: Dict[str, Any]
) -> str:
    """
    Build the LLM prompt for payload mapping.
    
    Args:
        tool_name: Name of the MCP tool
        resolved_schema: Schema after $ref resolution
        extracted_fields: Fields extracted from document
        sample_payload: Sample values loaded from JSON file
        
    Returns:
        Formatted prompt string
    """
    # Format schema in readable way
    schema_description = format_schema_for_llm(resolved_schema)
    
    return PAYLOAD_MAPPING_PROMPT.format(
        tool_name=tool_name,
        schema_description=schema_description,
        extracted_fields=json.dumps(extracted_fields, indent=2),
        sample_payload=json.dumps(sample_payload, indent=2)
    )


def format_schema_for_llm(schema: Dict[str, Any], indent: int = 0) -> str:
    """
    Format resolved schema in readable way for LLM.
    
    Args:
        schema: Resolved schema (no $refs)
        indent: Current indentation level
        
    Returns:
        Formatted string representation
    """
    if not isinstance(schema, dict):
        return str(schema)
    
    schema_type = schema.get('type', 'object')
    
    if schema_type == 'object' and 'properties' in schema:
        return format_object_schema(schema['properties'], indent)
    
    elif schema_type == 'array' and 'items' in schema:
        items = schema['items']
        if items.get('type') == 'object':
            return f"array of objects:\n{format_object_schema(items.get('properties', {}), indent + 1)}"
        else:
            return f"array of {items.get('type', 'unknown')}"
    
    else:
        return schema_type


def format_object_schema(properties: Dict[str, Any], indent: int = 0) -> str:
    """Format object properties."""
    lines = []
    prefix = "  " * indent
    
    for prop_name, prop_schema in properties.items():
        if not isinstance(prop_schema, dict):
            lines.append(f"{prefix}- {prop_name}: unknown")
            continue
        
        prop_type = prop_schema.get('type', 'string')
        prop_format = prop_schema.get('format')
        
        if prop_type == 'object':
            lines.append(f"{prefix}- {prop_name}: object {{")
            nested = format_object_schema(
                prop_schema.get('properties', {}),
                indent + 1
            )
            lines.append(nested)
            lines.append(f"{prefix}  }}")
        
        elif prop_type == 'array':
            items = prop_schema.get('items', {})
            item_type = items.get('type', 'unknown')
            
            if item_type == 'object':
                lines.append(f"{prefix}- {prop_name}: array of objects [")
                nested = format_object_schema(
                    items.get('properties', {}),
                    indent + 1
                )
                lines.append(nested)
                lines.append(f"{prefix}  ]")
            else:
                lines.append(f"{prefix}- {prop_name}: array of {item_type}")
        
        else:
            type_str = f"{prop_type}:{prop_format}" if prop_format else prop_type
            description = prop_schema.get('description', '')
            if description:
                lines.append(f"{prefix}- {prop_name}: {type_str}  # {description}")
            else:
                lines.append(f"{prefix}- {prop_name}: {type_str}")
    
    return "\n".join(lines)
