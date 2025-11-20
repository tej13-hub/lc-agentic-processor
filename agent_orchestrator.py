
"""
Fully Agentic Orchestrator with Document Splitting and Intelligent Routing + POST Submission (merged)

This file preserves the original logic from `agent_orchestrator1.py` and **adds** the
POST submission and structured logging features present in `agent_orchestrator.py`.
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from src.agents.preprocessing_agent import PreprocessingAgent
from src.agents.ocr_agent import OCRAgent
from src.agents.classifier_agent import ClassifierAgent
from src.agents.router_agent import RouterAgent
from src.agents.splitter_agent import SplitterAgent

# NOTE: Original file (`agent_orchestrator1.py`) uses create_llm_client via factory.
# Keep this intact.
from src.llm.llama_client import LlamaClient  # (kept as in original, though not used directly)
from src.llm.llm_factory import create_llm_client

from src.ocr.ocr_factory import create_ocr_engine
from src.utils.file_handler import FileHandler
from config.settings import settings

# *** ADDITIONS from second file ***
from src.agents.post_agent import PostAgent  # NEW
from src.logging.structured_logger import StructuredLogger  # NEW

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Fully agentic orchestrator with intelligent document routing + POST submission."""

    def __init__(self):
        """Initialize all autonomous agents."""
        logger.info("\n" + "=" * 70)
        logger.info("INITIALIZING MULTI-AGENT SYSTEM WITH INTELLIGENT ROUTING")
        logger.info("=" * 70)

        # Initialize file handler
        self.file_handler = FileHandler()

        # Initialize LLM client (shared)
        logger.info("\nInitializing LLM client...")
        # self.llm_client = LlamaClient(
        #     api_url=settings.LLAMA_API_URL,
        #     model=settings.LLAMA_MODEL,
        #     temperature=settings.LLM_TEMPERATURE,
        #     timeout=settings.LLM_TIMEOUT
        # )
        # Keep factory-based initialization as in original file
        self.llm_client = create_llm_client()

        # Initialize OCR engine
        logger.info("Initializing OCR engine...")
        self.ocr_engine = create_ocr_engine()

        # Initialize autonomous agents
        logger.info("\nInitializing autonomous agents...")
        self.splitter_agent = SplitterAgent(self.llm_client, self.ocr_engine)
        self.preprocessing_agent = PreprocessingAgent(self.llm_client)
        self.ocr_agent = OCRAgent(self.llm_client, self.ocr_engine)

        # Configure OCR agent (kept from original)
        self.ocr_agent.enable_llm_validation = settings.OCR_LLM_VALIDATION
        self.ocr_agent.confidence_threshold = settings.OCR_CONFIDENCE_THRESHOLD
        self.ocr_agent.min_text_length = settings.OCR_MIN_TEXT_LENGTH

        self.classifier_agent = ClassifierAgent(self.llm_client)
        self.router_agent = RouterAgent(self.llm_client)  # NEW: Router instead of generic extractor (kept)

        # === ADDITIONS: POST Agent + Structured Logger (optional) ===
        if getattr(settings, 'POST_ENABLED', False):
            self.post_agent = PostAgent(self.llm_client)
            logger.info("\u2713 POST agent initialized")
        else:
            self.post_agent = None
            logger.info("\u2139 POST agent disabled")

        self.structured_logger = StructuredLogger()  # NEW

        logger.info("\n" + "=" * 70)
        logger.info("\u2713 ALL AGENTS INITIALIZED")
        logger.info("=" * 70 + "\n")

    def split_document_if_needed(self, input_path: str) -> List[Dict[str, Any]]:
        """Split multi-page PDF into individual documents if needed."""
        is_valid, file_type = self.file_handler.validate_file(input_path)
        if not is_valid:
            raise ValueError("Invalid file")

        filename = Path(input_path).stem

        if file_type == 'pdf':
            pages = self.file_handler.pdf_to_images(input_path)
            if not pages:
                raise ValueError("Failed to load PDF pages")
            logger.info(f"PDF loaded: {len(pages)} page(s)")
            documents = self.splitter_agent.split_pdf_pages(pages, filename)
        else:
            logger.info(f"Image file loaded (single document)")
            image = self.file_handler.load_image(input_path)
            documents = [{
                'source': filename,
                'pages': [0],
                'page_range': '1',
                'document_id': f"{filename}_doc_001",
                'images': [image],
                'split_method': 'single_image'
            }]
        return documents

    def process_document(self, document: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Process a single document through the pipeline."""
        doc_id = document['document_id']
        page_range = document['page_range']
        images = document['images']

        logger.info("\n" + "=" * 70)
        logger.info(f"PROCESSING: {doc_id} (Pages: {page_range})")
        logger.info("=" * 70)
        print("\n" + "=" * 70)
        print(f"PROCESSING: {doc_id} (Pages: {page_range})")
        print("=" * 70)
        sys.stdout.flush()

        results: Dict[str, Any] = {
            "document_id": doc_id,
            "source_file": document['source'],
            "page_range": page_range,
            "status": "processing",
            "agent_decisions": {}
        }

        try:
            # Merge multi-page documents
            if len(images) > 1:
                logger.info(f"Merging {len(images)} pages...")
                print(f"Merging {len(images)} pages...")
                sys.stdout.flush()
                image = self.splitter_agent.merge_document_pages(images)
            else:
                image = images[0]

            # === AGENT 1: PREPROCESSING ===
            logger.info("\n" + "=" * 70)
            logger.info("AGENT 1: PREPROCESSING")
            logger.info("=" * 70)
            print("\n" + "=" * 70)
            print("AGENT 1: PREPROCESSING")
            print("=" * 70)
            sys.stdout.flush()

            preprocessed, preprocessing_decisions = self.preprocessing_agent.process(image)
            preprocessed_path = f"{output_dir}/{doc_id}_preprocessed.png"
            self.file_handler.save_image(preprocessed, preprocessed_path)
            results["agent_decisions"]["preprocessing"] = {
                "decisions": preprocessing_decisions,
                "output": preprocessed_path
            }
            print("\u2713 Preprocessing complete")
            sys.stdout.flush()

            # === AGENT 2: OCR ===
            logger.info("\n" + "=" * 70)
            logger.info("AGENT 2: OCR")
            logger.info("=" * 70)
            print("\n" + "=" * 70)
            print("AGENT 2: OCR")
            print("=" * 70)
            sys.stdout.flush()

            ocr_results = self.ocr_agent.extract_and_validate(preprocessed)
            validated_text = ocr_results['validated_text']
            logger.info(f"\nOCR Results Summary:")
            logger.info(f" Text length: {len(validated_text)}")
            logger.info(f" Confidence: {ocr_results.get('confidence', 0):.2%}")
            print(f"\u2713 OCR: Extracted {len(validated_text)} characters")
            print(f" Confidence: {ocr_results.get('confidence', 0):.2%}")
            sys.stdout.flush()

            if not validated_text or len(validated_text.strip()) < 10:
                logger.error("\u2717 CRITICAL: OCR extracted no meaningful text!")
                print("\u2717 ERROR: OCR extracted no meaningful text!")
                ocr_path = f"{output_dir}/{doc_id}_ocr.txt"
                with open(ocr_path, 'w', encoding='utf-8') as f:
                    f.write(f"OCR FAILED - No text extracted\n")
                    f.write(f"Confidence: {ocr_results.get('confidence', 0):.2%}\n")
                raise ValueError("OCR extracted no meaningful text")

            # Save OCR text
            ocr_path = f"{output_dir}/{doc_id}_ocr.txt"
            with open(ocr_path, 'w', encoding='utf-8') as f:
                f.write(validated_text)
            results["agent_decisions"]["ocr"] = {
                "confidence": ocr_results['confidence'],
                "text_length": ocr_results['text_length'],
                "llm_validation": ocr_results.get('llm_validation', 'unknown'),
                "output": ocr_path
            }

            # === AGENT 3: CLASSIFIER ===
            logger.info("\n" + "=" * 70)
            logger.info("AGENT 3: CLASSIFIER")
            logger.info("=" * 70)
            print("\n" + "=" * 70)
            print("AGENT 3: CLASSIFIER")
            print("=" * 70)
            sys.stdout.flush()

            classification = self.classifier_agent.classify(validated_text)
            raw_doc_type = classification.get('document_type')
            logger.info(f"\u2713 Classified as: {raw_doc_type}")
            logger.info(f" Confidence: {classification.get('document_confidence', 0):.2%}")
            print(f"\u2713 Classified as: {raw_doc_type}")
            print(f" Confidence: {classification.get('document_confidence', 0):.2%}")
            sys.stdout.flush()
            results["agent_decisions"]["classification"] = classification

            # === AGENT 4: ROUTER & EXTRACTION (NEW) ===
            logger.info("\n" + "=" * 70)
            logger.info("AGENT 4: ROUTER & EXTRACTOR")
            logger.info("=" * 70)
            print("\n" + "=" * 70)
            print("AGENT 4: ROUTER & EXTRACTOR")
            print("=" * 70)
            sys.stdout.flush()

            # Validate and normalize document type
            doc_type = self.router_agent.normalize_document_type(raw_doc_type)
            if doc_type != raw_doc_type:
                logger.warning(f"Document type normalized: {raw_doc_type} → {doc_type}")
                print(f"⚠ Type normalized: {raw_doc_type} → {doc_type}")
                sys.stdout.flush()

            # Check if extraction needed
            needs_extraction = self.router_agent.should_extract(doc_type)
            if needs_extraction:
                print(f"\u2713 Extraction required for {doc_type}")
                sys.stdout.flush()
                # Route to specific extractor and extract
                extracted_data = self.router_agent.route_and_extract(doc_type, validated_text)
                if extracted_data:
                    non_null_fields = sum(1 for v in extracted_data.values() if v is not None and v != "" and v != "null")
                    print(f"\u2713 Extracted {non_null_fields} fields")
                    sys.stdout.flush()
                else:
                    print(f"⚠ Extraction returned no data")
                    sys.stdout.flush()
            else:
                print(f"ℹ No extraction needed for {doc_type} (classification only)")
                sys.stdout.flush()
                extracted_data = None

            results["agent_decisions"]["extraction"] = extracted_data

            # === AGENT 5: POST SUBMISSION (ADDED) ===
            post_result = {}
            if self.post_agent and extracted_data is not None and extracted_data != "No extraction needed":
                logger.info("\n" + "=" * 70)
                logger.info("AGENT 5: POST SUBMISSION")
                logger.info("=" * 70)
                print("\n" + "=" * 70)
                print("AGENT 5: POST SUBMISSION")
                print("=" * 70)
                sys.stdout.flush()

                post_result = self.post_agent.submit_document(
                    document_id=doc_id,
                    document_type=doc_type,
                    extracted_fields=extracted_data
                )
                # Attach to results
                results['post_submission'] = post_result
                # if post_result.get('status') == 'success':
                #     logger.info("\n\u2713 Document submitted successfully!")
                #     print("\u2713 Document submitted successfully!")
                # else:
                #     logger.error("\n\u2717 Document submission failed")
                #     print("\u2717 Document submission failed")
                # sys.stdout.flush()
            else:
                # Ensure key exists even if not posted
                results['post_submission'] = {
                    'status': 'skipped',
                    'reason': 'Post agent disabled or no extracted data'
                }

            # Final results
            results["status"] = "success"
            results["final_output"] = {
                "instrument_type": "LC",
                "document_type": doc_type,
                "extracted_fields": extracted_data,
                # ADD post_status for downstream visibility
                "post_status": results["status"]
            }

            # Save results
            results_path = f"{output_dir}/{doc_id}_results.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            # Structured logging (ADDED)
            try:
                self.structured_logger.log_document_processing(
                    document_id=doc_id,
                    extraction=results.get('agent_decisions', {}),
                    post_result=results.get('post_submission', {})
                )
            except Exception:
                # Avoid breaking flow if structured logging fails
                logger.debug("Structured logging failed", exc_info=True)

            logger.info("\n" + "=" * 70)
            logger.info("\u2713 DOCUMENT PROCESSING COMPLETE")
            logger.info(f"\u2713 Results: {results_path}")
            logger.info("=" * 70 + "\n")
            print("\n" + "=" * 70)
            print("\u2713 DOCUMENT PROCESSING COMPLETE")
            print(f"\u2713 Results: {results_path}")
            print("=" * 70 + "\n")
            sys.stdout.flush()
            return results

        except Exception as e:
            logger.error(f"\n\u2717 Processing failed: {e}")
            print(f"\n\u2717 Processing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results["status"] = "failed"
            results["error"] = str(e)
            results["error_type"] = type(e).__name__
            results_path = f"{output_dir}/{doc_id}_results.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            return results

    def process_file(self, input_path: str, output_dir: str = None):
        """Process a file (may contain multiple documents)."""
        output_dir = output_dir or settings.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"\n{'#'*70}")
        logger.info(f"PROCESSING FILE: {Path(input_path).name}")
        logger.info(f"{'#'*70}")

        # Split into individual documents
        documents = self.split_document_if_needed(input_path)
        logger.info(f"\n\u2713 Found {len(documents)} document(s) in file")

        # Process each document
        all_results = []
        for i, doc in enumerate(documents, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"DOCUMENT {i}/{len(documents)}")
            logger.info(f"{'='*70}")
            result = self.process_document(doc, output_dir)
            all_results.append(result)
        return all_results

    def process_batch(self, input_dir: str, output_dir: str = None):
        """Process all files in input directory."""
        output_dir = output_dir or settings.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        # Find all supported files
        files = [f for f in os.listdir(input_dir)
                 if f.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'))]

        logger.info(f"\n{'='*70}")
        logger.info(f"BATCH PROCESSING: {len(files)} file(s)")
        logger.info(f"{'='*70}\n")

        batch_results = []
        for i, filename in enumerate(files, 1):
            logger.info(f"\n{'#'*70}")
            logger.info(f"FILE {i}/{len(files)}: {filename}")
            logger.info(f"{'#'*70}")
            input_path = os.path.join(input_dir, filename)
            try:
                file_results = self.process_file(input_path, output_dir)
                batch_results.extend(file_results)
            except Exception as e:
                logger.error(f"Failed to process {filename}: {e}")
                batch_results.append({
                    "source_file": filename,
                    "status": "failed",
                    "error": str(e)
                })

        # Save batch summary
        summary_path = f"{output_dir}/batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)

        # Statistics (kept)
        total_docs = len(batch_results)
        successful = sum(1 for r in batch_results if r.get('status') == 'success')
        failed = total_docs - successful
        with_extraction = sum(1 for r in batch_results
                              if r.get('status') == 'success' and r.get('agent_decisions', {}).get('extraction') is not None)

        logger.info("\n" + "=" * 70)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info(f" Files processed: {len(files)}")
        logger.info(f" Total documents: {total_docs}")
        logger.info(f" ✓ Successful: {successful}")
        logger.info(f" ✗ Failed: {failed}")
        logger.info(f" With extraction: {with_extraction}")
        logger.info(f" Classification only: {successful - with_extraction}")
        logger.info(f" Summary: {summary_path}")
        logger.info("=" * 70 + "\n")
        return batch_results
