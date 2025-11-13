"""
Application Settings with MCP Configuration
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal, List


class Settings(BaseSettings):
    """Application configuration settings."""
    
    # LLM Configuration
    LLM_TYPE: Literal["local", "remote"] = Field(default="local")
    LLAMA_API_URL: str = Field(default="http://localhost:11434/api/generate")
    LLAMA_MODEL: str = Field(default="llama3.2:3b")
    
    # Remote LLM
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
    OCR_MIN_TEXT_LENGTH: int = Field(default=10)
    OCR_LLM_VALIDATION: bool = Field(default=True)
    ocr_language_list: List[str] = Field(default=["en"])
    OCR_GPU: bool = Field(default=False)
    TESSERACT_CMD: str = Field(default="")
    
    # MCP Configuration (NEW)
    MCP_SERVER_URL: str = Field(default="http://localhost:8000")
    MCP_ENABLED: bool = Field(default=True)
    MCP_TIMEOUT: int = Field(default=30)
    
    # POST Configuration (NEW)
    POST_ENABLED: bool = Field(default=True)
    POST_VALIDATION: bool = Field(default=True)
    ALLOW_SAMPLE_VALUES: bool = Field(default=True)
    
    # Paths
    INPUT_DIR: str = Field(default="input")
    OUTPUT_DIR: str = Field(default="output")
    LOG_DIR: str = Field(default="logs")
    
    # Pydantic v2 configuration
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore"
    }


# Global settings instance
settings = Settings()


# Validate configuration
def validate_config():
    """Validate configuration."""
    if settings.LLM_TYPE not in ["local", "remote"]:
        raise ValueError(f"Invalid LLM_TYPE: {settings.LLM_TYPE}")
    
    if settings.LLM_TYPE == "remote" and not settings.REMOTE_LLM_API_KEY:
        print("WARNING: LLM_TYPE is 'remote' but REMOTE_LLM_API_KEY is not set!")
    
    if settings.MCP_ENABLED and settings.POST_ENABLED:
        print(f"✓ MCP enabled: {settings.MCP_SERVER_URL}")


validate_config()
