"""
Data Extraction Agent
"""

import logging
from typing import Dict, Any
from datetime import datetime
from src.llm.llama_client import LlamaClient
from prompts.extractor_prompts import SYSTEM_PROMPT_EXTRACTOR, get_extraction_prompt

logger = logging.getLogger(__name__)


class ExtractorAgent:
    """Agent for data extraction using LLM."""
    
    def __init__(self, llm_client: LlamaClient):
        self.llm = llm_client
        self.name = "ExtractorAgent"
        logger.info(f"✓ {self.name} initialized")
    
    def extract(self, ocr_text: str, document_type: str) -> Dict[str, Any]:
        """
        Extract data from document.
        
        Args:
            ocr_text: Extracted text
            document_type: Document type from classifier
            
        Returns:
            dict: Extracted fields
        """
        logger.info(f"{self.name}: Extracting data from {document_type}...")
        
        try:
            prompt = get_extraction_prompt(ocr_text, document_type)
            result = self.llm.generate_json(prompt, SYSTEM_PROMPT_EXTRACTOR)
            
            # Add metadata
            result['document_type'] = document_type
            result['extraction_timestamp'] = datetime.now().isoformat()
            
            # Count non-null fields
            field_count = len([v for v in result.values() if v is not None and v != ""])
            logger.info(f"✓ Extracted {field_count} fields")
            
            return result
            
        except Exception as e:
            logger.error(f"{self.name} failed: {e}")
            return {
                "error": str(e),
                "document_type": document_type
            }
