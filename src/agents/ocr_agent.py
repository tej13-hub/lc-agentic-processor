"""
OCR Agent - Intelligent OCR with optional LLM validation
"""

import logging
from typing import Dict, Any
import numpy as np

from src.llm.llama_client import LlamaClient
from src.ocr.base_ocr import BaseOCREngine

logger = logging.getLogger(__name__)


class OCRAgent:
    """
    Intelligent OCR agent with LLM-based quality validation and correction.
    """
    
    def __init__(self, llm_client: LlamaClient, ocr_engine: BaseOCREngine):
        self.llm = llm_client
        self.ocr_engine = ocr_engine
        self.name = "OCRAgent"
        
        # Configuration
        self.enable_llm_validation = True  # ✅ Enabled by default
        self.confidence_threshold = 0.85   # Only validate if confidence < 85%
        self.min_text_length = 100         # Only validate if text > 100 chars
        
        logger.info(f"✓ {self.name} initialized")
        logger.info(f"  LLM validation: {self.enable_llm_validation}")
        logger.info(f"  Validation threshold: confidence < {self.confidence_threshold:.0%}")
    
    def extract_and_validate(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract text and optionally validate/correct with LLM.
        
        LLM validation is SMART:
        - Only runs if OCR confidence is low
        - Only runs if text is substantial (not empty)
        - Uses shorter timeout
        - Gracefully falls back to raw OCR if LLM fails
        
        Args:
            image: Preprocessed image
            
        Returns:
            dict: OCR results with optional LLM corrections
        """
        logger.info(f"\n{self.name}: Extracting text...")
        logger.info(f"  Image shape: {image.shape}, dtype: {image.dtype}")
        
        try:
            # Step 1: Extract text with OCR engine
            ocr_result = self.ocr_engine.extract_structured(image)
            
            raw_text = ocr_result.get('full_text', '')
            avg_confidence = ocr_result.get('average_confidence', 0.0)
            
            logger.info(f"  ✓ Extracted {len(raw_text)} characters")
            logger.info(f"  ✓ OCR confidence: {avg_confidence:.2%}")
            
            # Check if we got meaningful text
            if not raw_text or len(raw_text.strip()) < 10:
                logger.warning(f"  ⚠ Very little text extracted: {len(raw_text)} chars")
                return {
                    "raw_text": raw_text,
                    "validated_text": raw_text,
                    "confidence": avg_confidence,
                    "text_length": len(raw_text),
                    "ocr_details": ocr_result,
                    "llm_validation": "skipped_insufficient_text"
                }
            
            # Show preview
            preview = raw_text[:150].replace('\n', ' ')
            logger.info(f"  Text preview: {preview}...")
            
            # Step 2: Decide if LLM validation is needed
            should_validate = self._should_validate_with_llm(raw_text, avg_confidence)
            
            if should_validate and self.enable_llm_validation:
                logger.info(f"  Running LLM validation (confidence: {avg_confidence:.2%} < {self.confidence_threshold:.0%})...")
                
                try:
                    validated_text = self.validate_with_llm(raw_text, avg_confidence)
                    
                    # Compare changes
                    if validated_text != raw_text:
                        changes = abs(len(validated_text) - len(raw_text))
                        logger.info(f"  ✓ LLM made corrections ({changes} character difference)")
                    else:
                        logger.info(f"  ✓ LLM validation: no changes needed")
                    
                    validation_status = "completed"
                    
                except Exception as e:
                    logger.warning(f"  ⚠ LLM validation failed: {e}")
                    logger.info(f"  Using raw OCR text (fallback)")
                    validated_text = raw_text
                    validation_status = f"failed_{type(e).__name__}"
            else:
                # High confidence or validation disabled - skip LLM
                reason = "disabled" if not self.enable_llm_validation else f"high_confidence_{avg_confidence:.0%}"
                logger.info(f"  Skipping LLM validation ({reason})")
                validated_text = raw_text
                validation_status = f"skipped_{reason}"
            
            return {
                "raw_text": raw_text,
                "validated_text": validated_text,
                "confidence": avg_confidence,
                "text_length": len(validated_text),
                "ocr_details": ocr_result,
                "llm_validation": validation_status
            }
            
        except Exception as e:
            logger.error(f"  ✗ OCR extraction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                "raw_text": "",
                "validated_text": "",
                "confidence": 0.0,
                "text_length": 0,
                "error": str(e),
                "llm_validation": "error"
            }
    
    def _should_validate_with_llm(self, text: str, confidence: float) -> bool:
        """
        Decide if LLM validation is worthwhile.
        
        Args:
            text: OCR extracted text
            confidence: OCR confidence score
            
        Returns:
            bool: True if should validate with LLM
        """
        # Don't validate if text too short
        if len(text) < self.min_text_length:
            return False
        
        # Don't validate if confidence already high
        if confidence >= self.confidence_threshold:
            return False
        
        # Don't validate if text is mostly garbage (too many non-alphanumeric)
        alphanumeric = sum(c.isalnum() or c.isspace() for c in text)
        if alphanumeric / len(text) < 0.5:  # Less than 50% readable chars
            return False
        
        return True
    
    def validate_with_llm(self, raw_text: str, confidence: float) -> str:
        """
        Use LLM to validate and correct OCR errors.
        
        This is SMART:
        - Only fixes OBVIOUS errors
        - Doesn't add/remove content
        - Uses short timeout
        - Returns original if uncertain
        
        Args:
            raw_text: Raw OCR output
            confidence: OCR confidence score
            
        Returns:
            str: Validated/corrected text
        """
        system_prompt = """You are an OCR error correction specialist for trade finance documents.

Your task: Fix ONLY obvious OCR errors while preserving exact meaning and structure.

Common OCR errors to fix:
- Number confusion: O→0, I→1, S→5, B→8, Z→2
- Letter confusion: rn→m, cl→d, vv→w, ii→u
- Case errors: lowercase 'l'→uppercase 'I'
- Special characters: I→l (in context)

CRITICAL RULES:
1. Fix ONLY obvious OCR errors
2. Do NOT add information that's not there
3. Do NOT remove information
4. Do NOT rephrase or rewrite
5. Preserve formatting (line breaks, spacing)
6. If uncertain, keep original

Return ONLY the corrected text, nothing else."""

        # Limit text length to avoid timeout
        text_sample = raw_text[:2000] if len(raw_text) > 2000 else raw_text
        
        prompt = f"""Review this OCR output and fix obvious errors:

OCR CONFIDENCE: {confidence:.2%}

TEXT:
{text_sample}

Return the corrected text (or same text if no obvious errors)."""

        try:
            # Use shorter timeout for validation
            corrected = self.llm.generate(
                prompt, 
                system_prompt,
                timeout=120  # 2 minutes max
            )
            
            # Sanity check: corrected text shouldn't be drastically different
            if len(corrected) < len(text_sample) * 0.5:
                logger.warning(f"  LLM returned text too short, using original")
                return raw_text
            
            if len(corrected) > len(text_sample) * 2:
                logger.warning(f"  LLM returned text too long, using original")
                return raw_text
            
            return corrected.strip()
            
        except Exception as e:
            logger.warning(f"  LLM validation error: {e}")
            return raw_text
