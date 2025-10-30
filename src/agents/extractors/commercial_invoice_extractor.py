"""
Commercial Invoice Extractor
"""

from src.agents.extractors.base_extractor import BaseExtractor
from pathlib import Path


class CommercialInvoiceExtractor(BaseExtractor):
    """Extractor for Commercial Invoice documents."""
    
    def __init__(self, llm_client):
        config_path = Path(__file__).parent.parent.parent / 'config' / 'document_configs' / 'commercial_invoice.yaml'
        super().__init__(llm_client, str(config_path))
