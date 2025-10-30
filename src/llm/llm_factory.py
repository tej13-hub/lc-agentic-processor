"""
LLM Factory - Creates appropriate LLM client based on configuration
"""

import logging
from config.settings import settings

logger = logging.getLogger(__name__)


def create_llm_client():
    """
    Create and return appropriate LLM client based on configuration.
    
    Returns:
        LLM client instance (LlamaClient or RemoteLLMClient)
    """
    llm_type = settings.LLM_TYPE.lower()
    
    if llm_type == "local":
        logger.info("Creating Local LLM client (Ollama)...")
        from src.llm.llama_client import LlamaClient
        
        return LlamaClient(
            api_url=settings.LLAMA_API_URL,
            model=settings.LLAMA_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            timeout=settings.LLM_TIMEOUT
        )
    
    elif llm_type == "remote":
        logger.info("Creating Remote LLM client (API)...")
        from src.llm.remote_llm_client import RemoteLLMClient
        
        if not settings.REMOTE_LLM_API_KEY:
            logger.warning("REMOTE_LLM_API_KEY is not set! API calls may fail.")
        
        return RemoteLLMClient(
            api_url=settings.REMOTE_LLM_API_URL,
            api_key=settings.REMOTE_LLM_API_KEY,
            model=settings.REMOTE_LLM_MODEL,
            temperature=settings.REMOTE_LLM_TEMPERATURE,
            timeout=settings.REMOTE_LLM_TIMEOUT
        )
    
    else:
        raise ValueError(f"Invalid LLM_TYPE: {llm_type}. Must be 'local' or 'remote'")
