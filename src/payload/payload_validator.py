"""
Payload Validator - Validate nested structures
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class PayloadValidator:
    """Validate payload against resolved schema - supports nested structures."""
    
    def __init__(self, resolved_schema: dict):
        """
        Initialize validator with resolved schema.
        
        Args:
            resolved_schema: Schema after $ref resolution
        """
        self.schema = resolved_schema
        
        # Extract all expected fields
        self.all_fields = self._get_all_field_names(resolved_schema)
        
        logger.debug(f"Validator initialized: {len(self.all_fields)} fields expected")
    
    def _get_all_field_names(self, schema: Dict[str, Any], prefix: str = "") -> List[str]:
        """Get all field names including nested."""
        fields = []
        
        if schema.get('type') == 'object' and 'properties' in schema:
            for prop_name, prop_schema in schema['properties'].items():
                current_path = f"{prefix}{prop_name}" if prefix else prop_name
                fields.append(current_path)
                
                if isinstance(prop_schema, dict):
                    if prop_schema.get('type') == 'object':
                        nested_fields = self._get_all_field_names(
                            prop_schema,
                            f"{current_path}."
                        )
                        fields.extend(nested_fields)
        
        return fields
    
    def validate(self, payload: dict, extracted_fields: dict, sample_payload: dict) -> dict:
        """
        Validate payload against schema.
        
        Args:
            payload: Payload to validate
            extracted_fields: Original extracted data
            sample_payload: Sample values
            
        Returns:
            Validation result: {valid: bool, errors: list, warnings: list}
        """
        errors = []
        warnings = []
        
        logger.info("Validating payload...")
        
        # Validate structure recursively
        structure_errors = self._validate_structure(
            payload,
            self.schema,
            path=""
        )
        errors.extend(structure_errors)
        
        # Track field sources
        fields_from_doc = []
        fields_from_sample = []
        
        self._track_field_sources(
            payload,
            extracted_fields,
            sample_payload,
            fields_from_doc,
            fields_from_sample,
            path=""
        )
        
        # Sanity checks
        sanity_warnings = self._sanity_checks(payload)
        warnings.extend(sanity_warnings)
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info(f"✓ Validation passed ({len(warnings)} warnings)")
            logger.info(f"  Fields from document: {len(fields_from_doc)}")
            logger.info(f"  Fields from samples: {len(fields_from_sample)}")
        else:
            logger.error(f"✗ Validation failed ({len(errors)} errors)")
            for error in errors[:5]:  # Show first 5 errors
                logger.error(f"  - {error}")
        
        return {
            'valid': is_valid,
            'errors': errors,
            'warnings': warnings,
            'fields_from_doc': fields_from_doc,
            'fields_from_sample': fields_from_sample,
            'total_fields': len(self.all_fields)
        }
    
    def _validate_structure(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str
    ) -> List[str]:
        """Recursively validate structure against schema."""
        errors = []
        
        if not isinstance(schema, dict):
            return errors
        
        schema_type = schema.get('type')
        
        if schema_type == 'object':
            if not isinstance(data, dict):
                errors.append(f"{path or 'root'} should be object, got {type(data).__name__}")
                return errors
            
            # Check all properties
            properties = schema.get('properties', {})
            for prop_name, prop_schema in properties.items():
                current_path = f"{path}.{prop_name}" if path else prop_name
                
                if prop_name not in data:
                    errors.append(f"Missing field: {current_path}")
                elif data[prop_name] is None or data[prop_name] == '':
                    errors.append(f"Empty field: {current_path}")
                else:
                    # Recursively validate nested
                    nested_errors = self._validate_structure(
                        data[prop_name],
                        prop_schema,
                        current_path
                    )
                    errors.extend(nested_errors)
        
        elif schema_type == 'array':
            if not isinstance(data, list):
                errors.append(f"{path} should be array, got {type(data).__name__}")
                return errors
            
            if len(data) == 0:
                errors.append(f"{path} array is empty")
            else:
                # Validate each item
                items_schema = schema.get('items', {})
                for i, item in enumerate(data):
                    item_path = f"{path}[{i}]"
                    item_errors = self._validate_structure(
                        item,
                        items_schema,
                        item_path
                    )
                    errors.extend(item_errors)
        
        elif schema_type in ['string', 'number', 'integer', 'boolean']:
            # Validate simple types
            if not self._validate_simple_type(data, schema, path):
                expected = schema_type
                actual = type(data).__name__
                errors.append(f"{path}: expected {expected}, got {actual}")
        
        return errors
    
    def _validate_simple_type(self, value: Any, schema: Dict[str, Any], path: str) -> bool:
        """Validate simple type."""
        schema_type = schema.get('type')
        
        if schema_type == 'string':
            if not isinstance(value, str):
                return False
            
            # Check format
            format_type = schema.get('format')
            if format_type == 'date':
                return self._validate_date_format(value)
        
        elif schema_type in ['number', 'integer']:
            if not isinstance(value, (int, float)):
                return False
        
        elif schema_type == 'boolean':
            if not isinstance(value, bool):
                return False
        
        return True
    
    def _validate_date_format(self, date_str: str) -> bool:
        """Validate date is in YYYY-MM-DD format."""
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except:
            return False
    
    def _track_field_sources(
        self,
        payload: Any,
        extracted: Any,
        sample: Any,
        from_doc: List[str],
        from_sample: List[str],
        path: str
    ):
        """Track which fields came from document vs sample."""
        if isinstance(payload, dict) and isinstance(extracted, dict) and isinstance(sample, dict):
            for key, value in payload.items():
                current_path = f"{path}.{key}" if path else key
                
                if key in extracted and extracted[key]:
                    if isinstance(value, (dict, list)):
                        # Recursively track nested
                        self._track_field_sources(
                            value,
                            extracted[key],
                            sample.get(key, {}),
                            from_doc,
                            from_sample,
                            current_path
                        )
                    else:
                        from_doc.append(current_path)
                else:
                    from_sample.append(current_path)
    
    def _sanity_checks(self, payload: dict) -> List[str]:
        """Perform sanity checks on payload values."""
        warnings = []
        
        # Check amount is reasonable
        if 'amount' in payload and payload['amount'] is not None:
            try:
                amount = float(payload['amount'])
                if amount <= 0:
                    warnings.append(f"Amount {amount} is <= 0")
                elif amount > 10000000:
                    warnings.append(f"Amount {amount} is unusually large (> 10M)")
            except:
                pass
        
        # Check date is not in future
        if 'date' in payload and payload['date']:
            try:
                date_obj = datetime.strptime(payload['date'], '%Y-%m-%d')
                if date_obj > datetime.now():
                    warnings.append(f"Date {payload['date']} is in the future")
            except:
                pass
        
        return warnings
