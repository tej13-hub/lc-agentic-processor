"""
Remote LLM Client for API-based LLMs (OpenAI, Azure, etc.)
"""

import requests
import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class RemoteLLMClient:
    """Client for interacting with remote LLM APIs."""
    
    def __init__(self, api_url: str, api_key: str, model: str, temperature: float = 0.1, timeout: int = 60):
        """
        Initialize Remote LLM client.
        
        Args:
            api_url: Remote API endpoint
            api_key: API authentication key
            model: Model name (e.g., 'gpt-4', 'claude-3-opus')
            temperature: Temperature for generation (0.0-1.0)
            timeout: Request timeout in seconds
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.default_timeout = timeout
        
        logger.info(f"RemoteLLMClient: {model} @ {api_url} (timeout: {timeout}s)")
    
    def generate(self, prompt: str, system_prompt: str = None, timeout: int = None) -> str:
        """
        Generate text response from remote LLM.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            timeout: Request timeout in seconds
            
        Returns:
            str: Generated text response
        """
        timeout = timeout or self.default_timeout
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Build request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 4096
        }
        
        # Build headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            logger.debug(f"Calling remote LLM (timeout: {timeout}s)...")
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract response text (OpenAI format)
            if "choices" in result and len(result["choices"]) > 0:
                generated_text = result["choices"][0]["message"]["content"]
            else:
                raise ValueError(f"Unexpected API response format: {result}")
            
            logger.debug(f"Remote LLM response length: {len(generated_text)} characters")
            
            return generated_text
            
        except requests.exceptions.Timeout:
            logger.error(f"Remote LLM request timed out after {timeout}s")
            raise TimeoutError(f"Remote LLM request timed out after {timeout} seconds")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Remote LLM request failed: {e}")
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error in remote LLM generation: {e}")
            raise
    
    def generate_json(self, prompt: str, system_prompt: str = None, timeout: int = None) -> dict:
        """
        Generate JSON response from remote LLM with robust parsing.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            timeout: Request timeout in seconds
            
        Returns:
            dict: Parsed JSON response
        """
        # Get text response
        response_text = self.generate(prompt, system_prompt, timeout)
        
        # Try multiple extraction methods
        
        # Method 1: Direct JSON parsing
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass
        
        # Method 2: Extract from markdown code blocks
        backtick = chr(96)
        
        # Pattern for ``````
        pattern_json = backtick + backtick + backtick + r'json\s*(\{.*?\})\s*' + backtick + backtick + backtick
        json_match = re.search(pattern_json, response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Pattern for ``````
        pattern_code = backtick + backtick + backtick + r'\s*(\{.*?\})\s*' + backtick + backtick + backtick
        code_match = re.search(pattern_code, response_text, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Method 3: Brace matching
        extracted = self._extract_json_from_text(response_text)
        if extracted:
            return extracted
        
        # Method 4: Manual extraction
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
        
        return {
            "document_type": "other",
            "document_confidence": 0.0,
            "reasoning": "Failed to parse LLM response"
        }
    
    def _extract_json_from_text(self, text: str) -> Optional[dict]:
        """Extract JSON object from text using brace matching."""
        start = text.find('{')
        if start == -1:
            return None
        
        brace_count = 0
        for i, char in enumerate(text[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        return None
        return None
