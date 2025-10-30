"""
Base Document Extractor
All specific extractors inherit from this
"""

import logging
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class BaseExtractor:
    """Base class for all document-specific extractors."""
    
    def __init__(self, llm_client, config_path: str):
        """
        Initialize extractor with YAML configuration.
        
        Args:
            llm_client: LLM client for extraction
            config_path: Path to YAML config file
        """
        self.llm = llm_client
        self.config = self._load_config(config_path)
        self.document_type = self.config['document_type']
        self.fields = self.config['fields']
        self.extraction_prompt_template = self.config['extraction_prompt']
        
        logger.info(f"✓ Initialized {self.document_type} extractor")
        logger.info(f"  Fields to extract: {len(self.fields)}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.debug(f"Loaded config from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
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
        
        # Limit text length to avoid token limits
        text_sample = text[:4000] if len(text) > 4000 else text
        
        # Fill template
        prompt = self.extraction_prompt_template.format(
            text=text_sample
        )
        
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
                        # Remove currency symbols and commas
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
