"""
Application settings and configuration
"""

import os
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application configuration settings."""
    
    # LLM API Configuration
    LLAMA_API_URL: str = "http://localhost:11434/api/generate"
    LLAMA_MODEL: str = "llama3.2:3b"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048
    LLM_TIMEOUT: int = 300 
    
    # OCR Configuration
    OCR_ENGINE: str = "paddleocr"
    OCR_LANGUAGES: str = "en"  # Comma-separated: "en,hi,ar"
    OCR_GPU: bool = False
    OCR_LLM_VALIDATION: bool = True
    OCR_CONFIDENCE_THRESHOLD: float = 0.85
    OCR_MIN_TEXT_LENGTH: int = 100
    
    # Preprocessing
    TARGET_DPI: int = 300
    
    # Output
    OUTPUT_DIR: str = "output"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    @property
    def ocr_language_list(self) -> List[str]:
        """Get OCR languages as list."""
        return [lang.strip() for lang in self.OCR_LANGUAGES.split(',')]


settings = Settings()
