"""
Classifier Agent - Classifies documents into predefined types
"""

import logging
import json
from typing import Dict, Any

from src.llm.llama_client import LlamaClient
from prompts.classifier_prompts import SYSTEM_PROMPT, get_classification_prompt

logger = logging.getLogger(__name__)


class ClassifierAgent:
    """
    Autonomous agent that classifies documents into predefined categories.
    """
    
    def __init__(self, llm_client: LlamaClient):
        self.llm = llm_client
        self.name = "ClassifierAgent"
        logger.info(f"✓ {self.name} initialized")
    
    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify document text into one of the predefined types.
        
        Args:
            text: OCR extracted text
            
        Returns:
            dict: Classification result with document_type, confidence, reasoning
        """
        logger.info(f"\n{self.name}: Classifying document...")
        
        try:
            # Build classification prompt
            prompt = get_classification_prompt(text)
            
            # Call LLM
            result = self.llm.generate_json(
                prompt,
                SYSTEM_PROMPT,  # Changed from SYSTEM_PROMPT_CLASSIFIER
                timeout=120
            )
            
            # Validate result
            doc_type = result.get('document_type', 'other')
            confidence = result.get('document_confidence', 0.0)
            reasoning = result.get('reasoning', 'No reasoning provided')
            
            logger.info(f"✓ Classification complete")
            logger.info(f"  Document type: {doc_type}")
            logger.info(f"  Confidence: {confidence:.2%}")
            logger.debug(f"  Reasoning: {reasoning}")
            
            return {
                "document_type": doc_type,
                "document_confidence": float(confidence),
                "reasoning": reasoning
            }
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return fallback
            return {
                "document_type": "other",
                "document_confidence": 0.0,
                "reasoning": f"Classification failed: {str(e)}"
            }
