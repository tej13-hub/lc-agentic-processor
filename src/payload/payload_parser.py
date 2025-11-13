"""
Payload Parser - Extract JSON from LLM response
"""

import json
import re
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class PayloadParser:
    """Parse LLM response to extract payload JSON."""
    
    def parse(self, llm_response: str) -> Optional[Dict[str, Any]]:
        """
        Parse LLM response with multiple fallback strategies.
        
        Args:
            llm_response: Raw response from LLM
            
        Returns:
            Parsed payload dict or None if parsing fails
        """
        logger.debug("Attempting to parse LLM response...")
        
        # Strategy 1: Direct JSON parse
        payload = self._try_direct_json(llm_response)
        if payload:
            logger.debug("✓ Parsed using direct JSON")
            return payload
        
        # Strategy 2: Extract from markdown code blocks
        payload = self._extract_from_markdown(llm_response)
        if payload:
            logger.debug("✓ Parsed from markdown block")
            return payload
        
        # Strategy 3: Extract JSON object from text
        payload = self._extract_json_object(llm_response)
        if payload:
            logger.debug("✓ Extracted JSON object from text")
            return payload
        
        # All strategies failed
        logger.error("✗ Failed to parse LLM response")
        logger.error(f"Raw response: {llm_response[:500]}...")
        return None
    
    def _try_direct_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Try direct JSON parsing."""
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return None
    
    def _extract_from_markdown(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from markdown code blocks."""
        # Pattern: ``````
        pattern = r'``````'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        return None
    
    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON object from text using regex."""
        # Find {...} in text
        pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
        
        return None
