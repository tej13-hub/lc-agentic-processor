"""
JSON Schema $ref Resolver
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class SchemaResolver:
    """Resolve $ref references in JSON Schema."""
    
    def resolve(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve all $ref in schema.
        
        Args:
            schema: inputSchema from MCP (may contain $defs and $refs)
            
        Returns:
            Resolved schema with no $refs
        """
        defs = schema.get('$defs', {})
        
        logger.info(f"Resolving schema (found {len(defs)} definitions)")
        
        resolved = self._resolve_refs(schema, defs)
        
        logger.info("✓ Schema resolved")
        
        return resolved
    
    def _resolve_refs(self, schema: Dict[str, Any], defs: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively resolve $ref."""
        
        # If this node is a $ref, replace it
        if isinstance(schema, dict) and '$ref' in schema:
            ref_path = schema['$ref']
            
            # Extract definition name from path (e.g., "#/$defs/PartyInfo" -> "PartyInfo")
            if ref_path.startswith('#/$defs/'):
                def_name = ref_path.split('/')[-1]
                
                if def_name in defs:
                    # Replace with definition and recursively resolve
                    return self._resolve_refs(defs[def_name].copy(), defs)
                else:
                    logger.warning(f"Definition not found: {def_name}")
                    return schema
        
        # If object with properties, resolve each property
        if isinstance(schema, dict):
            if schema.get('type') == 'object' and 'properties' in schema:
                resolved_props = {}
                for prop_name, prop_schema in schema['properties'].items():
                    resolved_props[prop_name] = self._resolve_refs(prop_schema, defs)
                schema['properties'] = resolved_props
            
            # If array with items, resolve items
            if schema.get('type') == 'array' and 'items' in schema:
                schema['items'] = self._resolve_refs(schema['items'], defs)
        
        return schema
    
    def get_all_field_paths(self, schema: Dict[str, Any], prefix: str = "") -> list:
        """
        Get all field paths including nested fields.
        
        Args:
            schema: Resolved schema
            prefix: Current path prefix
            
        Returns:
            List of field paths (e.g., ["document_id", "parties.name", "line_items[].quantity"])
        """
        fields = []
        
        if schema.get('type') == 'object' and 'properties' in schema:
            for prop_name, prop_schema in schema['properties'].items():
                current_path = f"{prefix}{prop_name}" if prefix else prop_name
                
                if prop_schema.get('type') == 'object':
                    # Nested object
                    nested_fields = self.get_all_field_paths(
                        prop_schema,
                        f"{current_path}."
                    )
                    fields.extend(nested_fields)
                elif prop_schema.get('type') == 'array':
                    # Array field
                    fields.append(f"{current_path}[]")
                    # Also get item fields
                    if 'items' in prop_schema:
                        item_fields = self.get_all_field_paths(
                            prop_schema['items'],
                            f"{current_path}[]."
                        )
                        fields.extend(item_fields)
                else:
                    # Simple field
                    fields.append(current_path)
        
        return fields
