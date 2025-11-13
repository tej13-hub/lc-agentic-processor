"""
Structured Logger - JSON logging for debugging
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from config.settings import settings

logger = logging.getLogger(__name__)


class StructuredLogger:
    """Log structured data for easy debugging."""
    
    def __init__(self):
        """Initialize structured logger."""
        self.log_dir = Path(settings.LOG_DIR)
        self.log_dir.mkdir(exist_ok=True)
    
    def log_document_processing(
        self,
        document_id: str,
        extraction: dict,
        post_result: dict
    ):
        """
        Log complete document processing flow.
        
        Args:
            document_id: Document identifier
            extraction: Extraction results
            post_result: POST submission result
        """
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file = self.log_dir / f"{document_id}_{timestamp}.json"
        
        log_entry = {
            'document_id': document_id,
            'timestamp': datetime.now().isoformat(),
            'extraction': extraction,
            'post_submission': post_result
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=2)
        
        logger.info(f"✓ Detailed log saved: {log_file}")
