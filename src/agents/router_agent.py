"""
Router Agent - Routes documents to appropriate extractors
Updated to use DynamicExtractor (no individual extractor classes needed!)
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class RouterAgent:
    """Routes classified documents to appropriate extractor agents."""
    
    def __init__(self, llm_client):
        self.llm = llm_client
        self.name = "RouterAgent"
        self.extractors = {}
        self.registry = self._load_registry()
        self.valid_types = self._build_valid_types_list()
        self._initialize_extractors()
        
        extraction_count = sum(1 for doc in self.registry['documents'] if doc.get('extract', False))
        
        logger.info(f"✓ {self.name} initialized")
        logger.info(f"  Total document types: {len(self.registry['documents'])}")
        logger.info(f"  Extractors loaded: {len(self.extractors)}")
        logger.info(f"  Extraction-enabled types: {extraction_count}")
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load document registry from YAML."""
        try:
            registry_path = Path(__file__).parent.parent.parent / 'config' / 'document_registry.yaml'
            
            logger.debug(f"Looking for registry at: {registry_path}")
            
            if not registry_path.exists():
                raise FileNotFoundError(f"Registry file not found at: {registry_path}")
            
            with open(registry_path, 'r', encoding='utf-8') as f:
                registry = yaml.safe_load(f)
            
            logger.info(f"✓ Loaded registry from {registry_path}")
            logger.debug(f"  Found {len(registry['documents'])} document types")
            return registry
            
        except Exception as e:
            logger.error(f"Failed to load document registry: {e}")
            logger.error(f"  Expected location: {Path(__file__).parent.parent.parent / 'config' / 'document_registry.yaml'}")
            raise
    
    def _build_valid_types_list(self) -> set:
        """Build set of valid document types from registry."""
        valid_types = {doc['type'] for doc in self.registry['documents']}
        logger.debug(f"Valid document types: {len(valid_types)}")
        return valid_types
    
    def _initialize_extractors(self):
        """
        Initialize all extractors using DynamicExtractor.
        No need for individual extractor classes!
        """
        
        from src.agents.extractors.dynamic_extractor import DynamicExtractor
        
        for doc in self.registry['documents']:
            if doc.get('extract', False):
                doc_type = doc['type']
                
                try:
                    # Use DynamicExtractor for ALL document types
                    self.extractors[doc_type] = DynamicExtractor(self.llm, doc_type)
                    
                    logger.info(f"  ✓ Loaded {doc_type} extractor (dynamic)")
                    
                except Exception as e:
                    logger.error(f"  ✗ Failed to load {doc_type} extractor: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
    
    def is_valid_document_type(self, doc_type: str) -> bool:
        """
        Check if document type exists in registry.
        
        Args:
            doc_type: Document type string
            
        Returns:
            bool: True if valid document type
        """
        return doc_type in self.valid_types
    
    def should_extract(self, doc_type: str) -> bool:
        """
        Check if document type requires extraction.
        
        Args:
            doc_type: Document type string
            
        Returns:
            bool: True if extraction required
        """
        for doc in self.registry['documents']:
            if doc['type'] == doc_type:
                return doc.get('extract', False)
        return False
    
    def get_extractor(self, doc_type: str):
        """
        Get specific extractor for document type.
        
        Args:
            doc_type: Document type string
            
        Returns:
            DynamicExtractor: Extractor instance or None
        """
        return self.extractors.get(doc_type)
    
    def normalize_document_type(self, doc_type: str) -> str:
        """
        Validate and normalize document type.
        Maps invalid types to 'other'.
        
        Args:
            doc_type: Raw document type from classifier
            
        Returns:
            str: Valid document type
        """
        if self.is_valid_document_type(doc_type):
            return doc_type
        else:
            logger.warning(f"Invalid document type '{doc_type}', mapping to 'other'")
            return "other"
    
    def route_and_extract(self, doc_type: str, text: str) -> Optional[Dict[str, Any]]:
        """
        Route document to appropriate extractor and extract fields.
        
        Args:
            doc_type: Classified document type
            text: OCR extracted text
            
        Returns:
            dict: Extracted fields, or None if extraction not needed
        """
        logger.info(f"\n{self.name}: Routing document type: {doc_type}")
        
        # Step 1: Validate document type
        normalized_type = self.normalize_document_type(doc_type)
        
        if normalized_type != doc_type:
            logger.info(f"  Document type normalized: {doc_type} → {normalized_type}")
            doc_type = normalized_type
        
        # Step 2: Check if extraction needed
        if not self.should_extract(doc_type):
            logger.info(f"  ℹ Extraction not required for {doc_type}")
            return None
        
        # Step 3: Get extractor
        extractor = self.get_extractor(doc_type)
        
        if extractor is None:
            logger.warning(f"  ⚠ No extractor found for {doc_type} (extraction enabled but extractor missing)")
            return None
        
        # Step 4: Extract
        logger.info(f"  ✓ Using DynamicExtractor for {doc_type}")
        result = extractor.extract(text)
        
        return result
