"""
Dynamic Extractor - Universal extractor that reads config from registry
No need for separate extractor classes!
"""

import logging
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class DynamicExtractor:
    """
    Universal extractor that works for ANY document type.
    Reads configuration directly from document registry.
    """
    
    def __init__(self, llm_client, doc_type: str):
        """
        Initialize extractor for any document type.
        
        Args:
            llm_client: LLM client for extraction
            doc_type: Type of document (e.g., 'airway_bill', 'cheque')
        """
        self.llm = llm_client
        self.document_type = doc_type
        self.config = self._load_config_from_registry(doc_type)
        self.fields = self.config.get('fields', [])
        self.extraction_prompt_template = self.config.get('extraction_prompt', '')
        
        logger.info(f"✓ DynamicExtractor initialized for {doc_type}")
        logger.info(f"  Fields to extract: {len(self.fields)}")
    
    def _load_config_from_registry(self, doc_type: str) -> Dict[str, Any]:
        """
        Load configuration for document type from registry.
        
        Args:
            doc_type: Document type to load
            
        Returns:
            dict: Configuration with fields and extraction prompt
        """
        try:
            # Path calculation:
            # __file__ = C:\...\lc-agentic-processor\src\agents\extractors\dynamic_extractor.py
            # parent = extractors/
            # parent.parent = agents/
            # parent.parent.parent = src/
            # parent.parent.parent.parent = PROJECT ROOT (lc-agentic-processor/)
            
            registry_path = Path(__file__).parent.parent.parent.parent / 'config' / 'document_registry.yaml'
            
            logger.debug(f"Looking for registry at: {registry_path}")
            
            if not registry_path.exists():
                raise FileNotFoundError(f"Registry not found at: {registry_path}")
            
            with open(registry_path, 'r', encoding='utf-8') as f:
                registry = yaml.safe_load(f)
            
            # Find document type in registry
            for doc in registry['documents']:
                if doc['type'] == doc_type:
                    if not doc.get('extract', False):
                        raise ValueError(f"Document type '{doc_type}' has extract=false in registry")
                    
                    logger.debug(f"Found config for {doc_type} in registry")
                    
                    return {
                        'document_type': doc_type,
                        'fields': doc.get('fields', []),
                        'extraction_prompt': doc.get('extraction_prompt', '')
                    }
            
            raise ValueError(f"Document type '{doc_type}' not found in registry")
            
        except Exception as e:
            logger.error(f"Failed to load config for {doc_type}: {e}")
            raise
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract fields from document text.
        
        Args:
            text: OCR extracted text
            
        Returns:
            dict: Extracted field values
        """
        logger.info(f"Extracting fields from {self.document_type}...")
        
        try:
            # Build prompt
            prompt = self._build_extraction_prompt(text)
            
            # Call LLM
            system_prompt = f"You are an expert at extracting structured data from {self.document_type} documents. Return ONLY valid JSON."
            
            logger.debug(f"Calling LLM for extraction...")
            result = self.llm.generate_json(
                prompt,
                system_prompt,
                timeout=180
            )

            # ADD THIS DEBUG
            print("\n" + "="*70)
            print("DEBUG: LLM RAW RESPONSE")
            print("="*70)
            print(f"Type: {type(result)}")
            print(f"Content: {result}")
            print("="*70 + "\n")
            
            # Validate and clean result
            validated_result = self._validate_extraction(result)
            
            non_null_count = sum(1 for v in validated_result.values() if v is not None and v != "")
            logger.info(f"✓ Extracted {non_null_count}/{len(self.fields)} fields")
            
            return validated_result
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._get_empty_result()
    
    def _build_extraction_prompt(self, text: str) -> str:
        """Build extraction prompt from template."""
        
        # Limit text length
        text_sample = text[:4000] if len(text) > 4000 else text
        
        # Fill template
        prompt = self.extraction_prompt_template.format(text=text_sample)
        
        return prompt
    
    def _validate_extraction(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean extracted data.
        
        Args:
            result: Raw extraction result from LLM
            
        Returns:
            dict: Validated and cleaned result
        """
        validated = {}
        
        for field in self.fields:
            field_name = field['name']
            value = result.get(field_name)
            
            # Convert "null" string to None
            if value == "null" or value == "None":
                value = None
            
            # Clean empty strings
            if isinstance(value, str) and value.strip() == "":
                value = None
            
            # Type-specific validation
            if value is not None:
                field_type = field.get('type', 'string')
                
                # Currency/number validation
                if field_type in ['currency', 'number']:
                    try:
                        if isinstance(value, str):
                            clean_value = value.replace(',', '').replace('$', '').replace('€', '').replace('₹', '').strip()
                            value = float(clean_value)
                        else:
                            value = float(value)
                    except (ValueError, TypeError):
                        logger.warning(f"  Could not convert '{field_name}' to number: {value}")
                        value = None
                
                # Date validation (basic)
                elif field_type == 'date':
                    if isinstance(value, str) and len(value) < 8:
                        logger.warning(f"  Invalid date format for '{field_name}': {value}")
                        value = None
            
            # Check required fields
            if field.get('required') and value is None:
                logger.warning(f"  Required field '{field_name}' is missing")
            
            validated[field_name] = value
        
        return validated
    
    def _get_empty_result(self) -> Dict[str, Any]:
        """Return empty result with all fields as null."""
        return {field['name']: None for field in self.fields}
