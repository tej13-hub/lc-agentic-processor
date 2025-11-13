"""
Agent Orchestrator with POST Submission
"""

import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from config.settings import settings
from src.agents.splitter_agent import SplitterAgent
from src.agents.preprocessing_agent import PreprocessingAgent
from src.agents.ocr_agent import OCRAgent
from src.agents.classifier_agent import ClassifierAgent
from src.agents.router_agent import RouterAgent
from src.agents.post_agent import PostAgent  # NEW
from src.logging.structured_logger import StructuredLogger  # NEW
from src.ocr.ocr_factory import create_ocr_engine
from src.llm.llm_factory import create_llm_client

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Orchestrates all agents including POST submission."""
    
    def __init__(self):
        """Initialize orchestrator with all agents."""
        
        logger.info("\n" + "="*70)
        logger.info("INITIALIZING MULTI-AGENT SYSTEM WITH POST")
        logger.info("="*70)
        
        # Initialize LLM client (shared)
        logger.info("\nInitializing LLM client...")
        self.llm_client = create_llm_client()
        
        # Initialize OCR engine
        logger.info("Initializing OCR engine...")
        self.ocr_engine = create_ocr_engine()
        
        # Initialize agents
        logger.info("Initializing agents...")
        self.splitter_agent = SplitterAgent()
        self.preprocessing_agent = PreprocessingAgent()
        self.ocr_agent = OCRAgent(self.ocr_engine)
        self.classifier_agent = ClassifierAgent(self.llm_client)
        self.router_agent = RouterAgent(self.llm_client)
        
        # NEW: POST agent
        if settings.POST_ENABLED:
            self.post_agent = PostAgent(self.llm_client)
            logger.info("✓ POST agent initialized")
        else:
            self.post_agent = None
            logger.info("ℹ POST agent disabled")
        
        # NEW: Structured logger
        self.structured_logger = StructuredLogger()
        
        logger.info("\n✓ All agents initialized successfully")
        logger.info("="*70 + "\n")
    
    def process_documents(self, input_dir: str, output_dir: str) -> List[Dict[str, Any]]:
        """
        Process all documents in input directory.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            
        Returns:
            List of processing results
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Find documents
        documents = self._find_documents(input_dir)
        
        if not documents:
            logger.warning(f"No documents found in {input_dir}")
            return []
        
        logger.info(f"\nFound {len(documents)} document(s)\n")
        
        # Process each document
        results = []
        for doc in documents:
            try:
                result = self.process_document(doc, output_dir)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {doc}: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info("\n" + "="*70)
        logger.info("  ✓ ALL PROCESSING COMPLETE")
        logger.info(f"  ✓ Check '{output_dir}/' for results")
        logger.info(f"  ✓ Check '{settings.LOG_DIR}/' for detailed logs")
        logger.info("="*70 + "\n")
        
        return results
    
    def process_document(self, document_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Process a single document through all agents.
        
        Args:
            document_path: Path to document
            output_dir: Output directory
            
        Returns:
            Processing result
        """
        logger.info("="*70)
        logger.info(f"PROCESSING: {Path(document_path).stem}")
        logger.info("="*70 + "\n")
        
        # Initialize result
        result = {
            'document_id': Path(document_path).stem,
            'source_file': Path(document_path).name,
            'status': 'processing',
            'agent_decisions': {},
            'post_submission': {},  # NEW
            'final_output': {}
        }
        
        try:
            # AGENT 1: Splitting
            logger.info("="*70)
            logger.info("AGENT 1: SPLITTING")
            logger.info("="*70)
            
            pages = self.splitter_agent.split(document_path)
            result['page_range'] = f"1-{len(pages)}" if len(pages) > 1 else "1"
            
            # Process first page
            current_image = pages[0]
            output_prefix = f"{output_dir}/{result['document_id']}"
            
            # AGENT 2: Preprocessing
            logger.info("\n" + "="*70)
            logger.info("AGENT 2: PREPROCESSING")
            logger.info("="*70)
            
            preprocessed_image = self.preprocessing_agent.preprocess(current_image, output_prefix)
            result['agent_decisions']['preprocessing'] = self.preprocessing_agent.get_decisions()
            
            # AGENT 3: OCR
            logger.info("\n" + "="*70)
            logger.info("AGENT 3: OCR")
            logger.info("="*70)
            
            ocr_result = self.ocr_agent.extract_text(preprocessed_image)
            result['agent_decisions']['ocr'] = {
                'confidence': ocr_result['confidence'],
                'text_length': len(ocr_result['text']),
                'output': f"{output_prefix}_ocr.txt"
            }
            
            # Validate OCR
            if not ocr_result['text'] or len(ocr_result['text']) < settings.OCR_MIN_TEXT_LENGTH:
                raise ValueError("OCR extracted no meaningful text")
            
            # AGENT 4: Classification
            logger.info("\n" + "="*70)
            logger.info("AGENT 4: CLASSIFIER")
            logger.info("="*70)
            
            classification = self.classifier_agent.classify(ocr_result['text'])
            result['agent_decisions']['classification'] = classification
            
            # AGENT 5: Router & Extraction
            logger.info("\n" + "="*70)
            logger.info("AGENT 5: ROUTER & EXTRACTOR")
            logger.info("="*70)
            
            extraction = self.router_agent.route_and_extract(
                classification['document_type'],
                ocr_result['text']
            )
            result['agent_decisions']['extraction'] = extraction
            
            # AGENT 6: POST Submission (NEW)
            if self.post_agent and extraction and extraction != "No extraction needed":
                logger.info("\n" + "="*70)
                logger.info("AGENT 6: POST SUBMISSION")
                logger.info("="*70)
                
                post_result = self.post_agent.submit_document(
                    document_id=result['document_id'],
                    document_type=classification['document_type'],
                    extracted_fields=extraction
                )
                
                result['post_submission'] = post_result
                
                if post_result['status'] == 'success':
                    logger.info("\n✓ Document submitted successfully!")
                else:
                    logger.error("\n✗ Document submission failed")
            
            # Final output
            result['status'] = 'success'
            result['final_output'] = {
                'document_type': classification['document_type'],
                'extracted_fields': extraction if extraction != "No extraction needed" else None,
                'post_status': result.get('post_submission', {}).get('status')
            }
            
            # Save result
            import json
            with open(f"{output_prefix}_results.json", 'w') as f:
                json.dump(result, f, indent=2)
            
            # Structured logging
            self.structured_logger.log_document_processing(
                document_id=result['document_id'],
                extraction=result['agent_decisions'],
                post_result=result.get('post_submission', {})
            )
            
            logger.info("\n" + "="*70)
            logger.info("✓ DOCUMENT PROCESSING COMPLETE")
            logger.info(f"✓ Results: {output_prefix}_results.json")
            logger.info("="*70 + "\n")
            
        except Exception as e:
            logger.error(f"\n✗ Processing failed: {e}")
            result['status'] = 'failed'
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()
        
        return result
    
    def _find_documents(self, input_dir: str) -> List[str]:
        """Find all documents in input directory."""
        supported_formats = ['.pdf', '.png', '.jpg', '.jpeg', '.tif', '.tiff']
        
        documents = []
        for ext in supported_formats:
            documents.extend(Path(input_dir).glob(f'*{ext}'))
        
        return [str(doc) for doc in documents]
