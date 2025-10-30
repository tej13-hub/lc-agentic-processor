"""
Demand Draft Extractor
"""

from src.agents.extractors.base_extractor import BaseExtractor
from pathlib import Path


class DemandDraftExtractor(BaseExtractor):
    """Extractor for Demand Draft documents."""
    
    def __init__(self, llm_client):
        config_path = Path(__file__).parent.parent.parent / 'config' / 'document_configs' / 'demand_draft.yaml'
        super().__init__(llm_client, str(config_path))
