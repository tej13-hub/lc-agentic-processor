"""
Application Settings
Loads configuration from environment variables
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal, List


class Settings(BaseSettings):
    """Application configuration settings."""
    
    # LLM Configuration
    LLM_TYPE: Literal["local", "remote"] = Field(default="local", description="LLM type: local or remote")
    
    # Local LLM (Ollama)
    LLAMA_API_URL: str = Field(default="http://localhost:11434/api/generate")
    LLAMA_MODEL: str = Field(default="llama3.2:3b")
    
    # Remote LLM (API-based)
    REMOTE_LLM_API_URL: str = Field(default="https://api.openai.com/v1/chat/completions")
    REMOTE_LLM_API_KEY: str = Field(default="")
    REMOTE_LLM_MODEL: str = Field(default="gpt-4")
    REMOTE_LLM_TEMPERATURE: float = Field(default=0.1)
    REMOTE_LLM_TIMEOUT: int = Field(default=60)
    
    # Common LLM Settings
    LLM_TEMPERATURE: float = Field(default=0.1)
    LLM_TIMEOUT: int = Field(default=300)
    
    # OCR Configuration
    OCR_ENGINE: str = Field(default="paddleocr")
    OCR_CONFIDENCE_THRESHOLD: float = Field(default=0.6)
    OCR_MIN_TEXT_LENGTH: int = Field(default=50)
    OCR_LLM_VALIDATION: bool = Field(default=True)
    ocr_language_list: List[str] = Field(default=["en"])
    OCR_GPU: bool = Field(default=False)
    
    # Paths
    INPUT_DIR: str = Field(default="input")
    OUTPUT_DIR: str = Field(default="output")
    
    # Pydantic v2 configuration
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore"
    }


# Global settings instance
settings = Settings()


# Validate LLM configuration on load
def validate_llm_config():
    """Validate LLM configuration based on type."""
    if settings.LLM_TYPE not in ["local", "remote"]:
        raise ValueError(f"Invalid LLM_TYPE: {settings.LLM_TYPE}. Must be 'local' or 'remote'")
    
    if settings.LLM_TYPE == "remote" and not settings.REMOTE_LLM_API_KEY:
        print("WARNING: LLM_TYPE is 'remote' but REMOTE_LLM_API_KEY is not set!")


# Run validation
validate_llm_config()
