"""
Cheque Extractor
"""

from src.agents.extractors.base_extractor import BaseExtractor
from pathlib import Path


class ChequeExtractor(BaseExtractor):
    """Extractor for Cheque documents."""
    
    def __init__(self, llm_client):
        config_path = Path(__file__).parent.parent.parent / 'config' / 'document_configs' / 'cheque.yaml'
        super().__init__(llm_client, str(config_path))
