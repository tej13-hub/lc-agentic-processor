"""
LLM Client for Llama integration via Ollama
"""

import requests
import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class LlamaClient:
    """Client for interacting with Llama models via Ollama API."""
    
    def __init__(self, api_url: str, model: str, temperature: float = 0.2, timeout: int = 300):
        """
        Initialize Llama client.
        
        Args:
            api_url: Ollama API endpoint
            model: Model name (e.g., 'llama3.2:3b')
            temperature: Temperature for generation (0.0-1.0)
            timeout: Default timeout in seconds
        """
        self.api_url = api_url
        self.model = model
        self.temperature = temperature
        self.default_timeout = timeout
        
        logger.info(f"LlamaClient: {model} @ {api_url} (timeout: {timeout}s)")
    
    def generate(self, prompt: str, system_prompt: str = None, timeout: int = None) -> str:
        """
        Generate text response from LLM.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            timeout: Request timeout in seconds (uses default if not specified)
            
        Returns:
            str: Generated text response
            
        Raises:
            TimeoutError: If request times out
            Exception: For other errors
        """
        # timeout = timeout or self.default_timeout
        timeout = 300
        
        # Build request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature
            }
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            logger.debug(f"Calling LLM (timeout: {timeout}s)...")
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get('response', '')
            
            logger.debug(f"LLM response length: {len(generated_text)} characters")
            
            return generated_text
            
        except requests.exceptions.Timeout:
            logger.error(f"LLM request timed out after {timeout}s")
            raise TimeoutError(f"LLM request timed out after {timeout} seconds")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM request failed: {e}")
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error in LLM generation: {e}")
            raise
    
    def generate_json(self, prompt: str, system_prompt: str = None, timeout: int = None) -> dict:
        """
        Generate JSON response from LLM with robust parsing.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            timeout: Request timeout in seconds
            
        Returns:
            dict: Parsed JSON response
        """
        # Get text response
        response_text = self.generate(prompt, system_prompt, timeout)
        
        # Method 1: Direct JSON parsing
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass
        
        # Method 2: Extract from markdown code blocks
        # Use chr(96) to avoid syntax issues with backticks
        backtick = chr(96)
        
        # Pattern for ``````
        pattern_json = backtick + backtick + backtick + r'json\s*(\{.*?\})\s*' + backtick + backtick + backtick
        json_match = re.search(pattern_json, response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Pattern for `````` (no language tag)
        pattern_code = backtick + backtick + backtick + r'\s*(\{.*?\})\s*' + backtick + backtick + backtick
        code_match = re.search(pattern_code, response_text, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Method 3: Brace matching (most reliable)
        extracted = self._extract_json_from_text(response_text)
        if extracted:
            return extracted
        
        # Method 4: Manual extraction of key-value pairs (fallback)
        try:
            doc_type_match = re.search(r'"document_type"\s*:\s*"([^"]+)"', response_text)
            confidence_match = re.search(r'"document_confidence"\s*:\s*([\d.]+)', response_text)
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', response_text)
            
            if doc_type_match:
                return {
                    "document_type": doc_type_match.group(1),
                    "document_confidence": float(confidence_match.group(1)) if confidence_match else 0.0,
                    "reasoning": reasoning_match.group(1) if reasoning_match else "Extracted from partial response"
                }
        except Exception:
            pass
        
        # All methods failed
        logger.error("JSON extraction failed: No valid JSON found in response")
        logger.debug(f"Raw response (first 500 chars): {response_text[:500]}")
        
        # Return safe fallback
        return {
            "document_type": "other",
            "document_confidence": 0.0,
            "reasoning": "Failed to parse LLM response"
        }
    
    def _extract_json_from_text(self, text: str) -> Optional[dict]:
        """
        Extract JSON object from text using brace matching.
        Handles nested JSON correctly.
        
        Args:
            text: Text containing JSON
            
        Returns:
            dict: Parsed JSON or None if not found
        """
        # Find first opening brace
        start = text.find('{')
        if start == -1:
            return None
        
        # Count braces to find matching closing brace
        brace_count = 0
        for i, char in enumerate(text[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found matching closing brace
                    json_str = text[start:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        # Try to find next opening brace
                        next_start = text.find('{', start + 1)
                        if next_start != -1:
                            return self._extract_json_from_text(text[next_start:])
                        return None
        
        return None
