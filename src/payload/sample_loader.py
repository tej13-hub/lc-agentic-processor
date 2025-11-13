"""
Sample Data Loader - Load from JSON files
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SampleLoader:
    """Load sample payloads from JSON files."""
    
    def __init__(self, samples_dir: str = "config/samples"):
        """
        Initialize sample loader.
        
        Args:
            samples_dir: Directory containing sample JSON files
        """
        self.samples_dir = Path(samples_dir)
        
        if not self.samples_dir.exists():
            logger.warning(f"Samples directory not found: {self.samples_dir}")
            self.samples_dir.mkdir(parents=True, exist_ok=True)
    
    def load_sample(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Load sample payload for a tool.
        
        Args:
            tool_name: Name of tool (e.g., "setDocumentDetails")
            
        Returns:
            Sample payload dict or None if not found
        """
        sample_file = self.samples_dir / f"{tool_name}.json"
        
        if not sample_file.exists():
            logger.warning(f"Sample file not found: {sample_file}")
            return None
        
        try:
            with open(sample_file, 'r') as f:
                sample_data = json.load(f)
            
            logger.info(f"✓ Loaded sample from: {sample_file}")
            logger.debug(f"  Sample has {len(sample_data)} top-level fields")
            
            return sample_data
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in sample file: {sample_file}")
            logger.error(f"  Error: {e}")
            return None
        
        except Exception as e:
            logger.error(f"Failed to load sample: {e}")
            return None
    
    def get_available_samples(self) -> list:
        """Get list of available sample files."""
        return [
            f.stem for f in self.samples_dir.glob('*.json')
        ]
