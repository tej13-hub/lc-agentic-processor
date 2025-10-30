"""
Bill of Exchange Extractor
"""

from src.agents.extractors.base_extractor import BaseExtractor
from pathlib import Path


class BillOfExchangeExtractor(BaseExtractor):
    """Extractor for Bill of Exchange documents."""
    
    def __init__(self, llm_client):
        config_path = Path(__file__).parent.parent.parent / 'config' / 'document_configs' / 'bill_of_exchange.yaml'
        super().__init__(llm_client, str(config_path))
