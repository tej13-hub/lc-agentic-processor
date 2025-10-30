"""
Main entry point for LC Agentic Document Processor
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging BEFORE any imports
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('lc_agentic_processor.log', mode='w', encoding='utf-8')
    ],
    force=True
)

# Suppress verbose third-party loggers
logging.getLogger('ppocr').setLevel(logging.ERROR)
logging.getLogger('paddlex').setLevel(logging.ERROR)
logging.getLogger('paddle').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Get logger for this module
logger = logging.getLogger(__name__)

# Now import the rest
from src.orchestrator.agent_orchestrator import AgentOrchestrator
from config.settings import settings


def main():
    """Main execution."""
    
    print("\n" + "="*70)
    print("  FULLY AGENTIC LC DOCUMENT PROCESSOR")
    print("="*70)
    
    logger.info("Starting LC Document Processor...")
    
    # Initialize the orchestrator
    orchestrator = AgentOrchestrator()
    
    # Setup directories
    input_dir = 'input'
    output_dir = settings.OUTPUT_DIR
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Find input files
    input_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'))]
    
    if not input_files:
        logger.warning("No documents found in input/ folder")
        print("\n⚠ No documents found in input/ folder")
        print("  Please add PDF or image files to the input/ folder\n")
        return
    
    logger.info(f"Found {len(input_files)} document(s)")
    print(f"\nFound {len(input_files)} document(s)")
    
    # Process batch
    logger.info("Starting batch processing...")
    orchestrator.process_batch(input_dir, output_dir)
    
    logger.info("Processing complete")
    print("\n" + "="*70)
    print("  ✓ ALL AUTONOMOUS AGENTS COMPLETED THEIR TASKS")
    print("  ✓ Check 'output/' for results")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        print("\n\n⚠ Processing interrupted by user\n")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n✗ Fatal error: {e}\n")
        sys.exit(1)
