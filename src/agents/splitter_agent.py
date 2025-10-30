"""
Document Splitter Agent - Intelligent multi-page document handling
Distinguishes between:
- Multi-page SINGLE documents (keep together)
- Multi-document PDFs (split intelligently)
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any
from PIL import Image

from src.llm.llama_client import LlamaClient
from src.ocr.easyocr_engine import EasyOCREngine

logger = logging.getLogger(__name__)


class SplitterAgent:
    """
    Intelligent document splitter using LLM.
    
    Key capability: Distinguishes between:
    - Single multi-page document (Invoice pages 1-3) → Keep as one
    - Multiple documents (Invoice + B/L + Certificate) → Split
    """
    
    def __init__(self, llm_client: LlamaClient, ocr_engine: EasyOCREngine):
        self.llm = llm_client
        self.ocr_engine = ocr_engine
        self.name = "SplitterAgent"
        logger.info(f"✓ {self.name} initialized")
    
    def split_pdf_pages(
        self, 
        pages: List[Image.Image],
        source_filename: str
    ) -> List[Dict[str, Any]]:
        """
        Intelligently split or keep together PDF pages.
        
        Args:
            pages: List of PIL Images (PDF pages)
            source_filename: Original filename
            
        Returns:
            List of document dictionaries
        """
        num_pages = len(pages)
        
        logger.info(f"\n{self.name}: Analyzing {num_pages} page(s) from {source_filename}...")
        
        # Single page - always one document
        if num_pages == 1:
            logger.info("  ✓ Single page, treating as one document")
            return self._create_single_document(pages, source_filename)
        
        # Multi-page: Extract text from all pages for analysis
        logger.info(f"  Extracting text from {num_pages} pages for intelligent analysis...")
        page_texts = self._extract_all_page_texts(pages)
        
        # Check if any pages have content
        pages_with_content = sum(1 for pt in page_texts if pt['has_content'])
        
        if pages_with_content == 0:
            logger.info(f"  ✓ All pages empty, treating as one document")
            return self._create_single_document(pages, source_filename)
        
        if pages_with_content == 1:
            logger.info(f"  ✓ Only one page has content, treating as one document")
            return self._create_single_document(pages, source_filename)
        
        # Use LLM for intelligent boundary detection
        logger.info(f"  Analyzing document structure with LLM...")
        analysis = self._analyze_document_structure(page_texts)
        
        # Check analysis result
        if analysis['is_single_document']:
            logger.info(f"  ✓ LLM determined: Single {analysis.get('document_type', 'document')} spanning {num_pages} pages")
            logger.info(f"  Reasoning: {analysis.get('reasoning', 'N/A')}")
            return self._create_single_document(pages, source_filename)
        
        # Multiple documents detected
        boundaries = analysis.get('boundaries', [0])
        num_docs = len(boundaries)
        
        logger.info(f"  ✓ LLM detected {num_docs} separate document(s)")
        logger.info(f"  Boundaries at pages: {[b+1 for b in boundaries]}")
        logger.info(f"  Reasoning: {analysis.get('reasoning', 'N/A')}")
        
        # Group pages into documents
        documents = self._group_pages_into_documents(
            pages, 
            boundaries, 
            source_filename,
            analysis
        )
        
        logger.info(f"✓ Result: {len(documents)} document(s)")
        return documents
    
    def _extract_all_page_texts(self, pages: List[Image.Image]) -> List[Dict[str, Any]]:
        """Extract text from all pages for analysis."""
        page_texts = []
        
        for i, page in enumerate(pages):
            logger.info(f"    Page {i+1}/{len(pages)}...")
            page_array = np.array(page)
            ocr_result = self.ocr_engine.extract_text(page_array)
            
            # Extract meaningful snippets
            text_clean = ocr_result.strip()
            has_content = len(text_clean) > 50
            
            # Get first and last portions for analysis
            first_200 = text_clean[:200] if text_clean else ""
            last_200 = text_clean[-200:] if len(text_clean) > 200 else ""
            
            page_texts.append({
                'page_number': i,
                'text_length': len(text_clean),
                'has_content': has_content,
                'first_200': first_200,
                'last_200': last_200,
                'full_text_sample': text_clean[:800]  # First 800 chars for LLM
            })
        
        return page_texts
    
    def _analyze_document_structure(
        self, 
        page_texts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Use LLM to analyze if pages form single document or multiple documents.
        
        This is the CORE intelligence of the splitter.
        """
        logger.info(f"  {self.name}: Performing intelligent document structure analysis...")
        
        # Build detailed page analysis for LLM
        pages_analysis = []
        
        for i, pt in enumerate(page_texts):
            if not pt['has_content']:
                pages_analysis.append(f"Page {i+1}: [Empty or minimal content]")
            else:
                page_info = f"""Page {i+1} ({pt['text_length']} chars):
  First 200 chars: {pt['first_200']}
  Last 200 chars: {pt['last_200']}"""
                pages_analysis.append(page_info)
        
        pages_text = "\n\n".join(pages_analysis)
        
        prompt = f"""You are analyzing a {len(page_texts)}-page PDF to determine its document structure.

CRITICAL TASK: Determine if this PDF contains:
A) ONE multi-page document (e.g., a 3-page invoice, a 5-page LC)
B) MULTIPLE separate documents (e.g., Invoice + Bill of Lading + Certificate)

ANALYSIS OF PAGES:
{pages_text}

DECISION CRITERIA:

SINGLE DOCUMENT indicators (keep all pages together):
- Continuation phrases: "continued on next page", "page 2 of 3", "(cont'd)"
- Same document number across pages (e.g., Invoice #123 on all pages)
- Same sender/receiver on all pages
- Consistent formatting and layout
- Progressive content (items listed, terms continuing, etc.)
- No new document headers on subsequent pages
- Last page has endings: "Total", "Signature", "Terms and Conditions"

MULTIPLE DOCUMENTS indicators (split):
- NEW document headers on different pages (e.g., "COMMERCIAL INVOICE" then "BILL OF LADING")
- Different document numbers (Invoice #123, then B/L #456)
- Different document types entirely
- Different sender/receiver pairs
- Clear start of new document (header, title, new formatting)

IMPORTANT EXAMPLES:

Example 1 - SINGLE 3-page invoice:
  Page 1: "COMMERCIAL INVOICE #12345, Items 1-20"
  Page 2: "Invoice #12345 (continued), Items 21-40"
  Page 3: "Invoice #12345 (continued), Total: $50,000, Signature"
  → SINGLE DOCUMENT (3 pages belong together)

Example 2 - MULTIPLE documents (Invoice + B/L):
  Page 1: "COMMERCIAL INVOICE #12345"
  Page 2: "BILL OF LADING #BL789" (NEW DOCUMENT)
  → MULTIPLE DOCUMENTS (split at page 2)

Example 3 - SINGLE 5-page LC:
  Page 1: "LETTER OF CREDIT LC987654, Terms and Conditions"
  Page 2-4: "LC987654 continued, Additional clauses"
  Page 5: "LC987654 end, Bank signatures"
  → SINGLE DOCUMENT (5 pages belong together)

YOUR TASK:
Analyze the pages above and return JSON:

{{
  "is_single_document": true/false,
  "document_type": "commercial_invoice" / "letter_of_credit" / "bill_of_lading" / "mixed" / "other",
  "num_documents": 1 or more,
  "boundaries": [0, 3, 5],
  "reasoning": "Detailed explanation of your decision",
  "confidence": 0.0-1.0
}}

EXPLANATION:
- "is_single_document": true if ALL pages are part of ONE document
- "boundaries": Page indices (0-indexed) where NEW documents START
  - For single document: [0] only
  - For multiple: [0, 3, 5] means doc1=pages 0-2, doc2=pages 3-4, doc3=page 5
- "reasoning": Clear explanation citing evidence from page analysis

BE CONSERVATIVE: If uncertain, treat as single document (avoid over-splitting).

Return ONLY valid JSON."""

        system_prompt = """You are an expert document analyst specializing in trade finance documents.

Your expertise:
- Recognizing multi-page document continuity
- Identifying true document boundaries
- Understanding LC, Invoice, B/L, Certificate structures
- Detecting pagination vs new documents

Core principle: Multi-page documents are common in trade finance. Don't split unless clear evidence of multiple documents.

Return ONLY JSON."""

        try:
            result = self.llm.generate_json(
                prompt, 
                system_prompt,
                timeout=240  # 4 minutes for thorough analysis
            )
            
            # Validate result
            is_single = result.get('is_single_document', True)  # Default to single
            boundaries = result.get('boundaries', [0])
            
            # Ensure boundaries is a list of integers
            if not isinstance(boundaries, list):
                boundaries = [0]
            else:
                # Convert to 0-indexed if needed, filter invalid values
                boundaries = [
                    int(b) if isinstance(b, int) and b == 0 else int(b) - 1 
                    for b in boundaries 
                    if isinstance(b, (int, float))
                ]
                boundaries = [b for b in boundaries if 0 <= b < len(page_texts)]
            
            # Ensure first page is a boundary
            if 0 not in boundaries:
                boundaries.insert(0, 0)
            
            # Sort and deduplicate
            boundaries = sorted(list(set(boundaries)))
            
            # Override: If only one boundary, it's a single document
            if len(boundaries) == 1:
                is_single = True
            
            return {
                'is_single_document': is_single,
                'document_type': result.get('document_type', 'unknown'),
                'num_documents': len(boundaries),
                'boundaries': boundaries,
                'reasoning': result.get('reasoning', 'LLM analysis completed'),
                'confidence': result.get('confidence', 0.8)
            }
            
        except Exception as e:
            logger.error(f"  LLM analysis failed: {e}")
            logger.warning("  Falling back to conservative approach (treat as single document)")
            return {
                'is_single_document': True,
                'document_type': 'unknown',
                'num_documents': 1,
                'boundaries': [0],
                'reasoning': f'LLM failed, conservative fallback: {str(e)}',
                'confidence': 0.5
            }
    
    def _create_single_document(
        self, 
        pages: List[Image.Image], 
        source_filename: str
    ) -> List[Dict[str, Any]]:
        """Create a single document entry (no splitting)."""
        page_indices = list(range(len(pages)))
        
        if len(pages) == 1:
            page_range = "1"
        else:
            page_range = f"1-{len(pages)}"
        
        return [{
            'source': source_filename,
            'pages': page_indices,
            'page_range': page_range,
            'document_id': f"{source_filename}_doc_001",
            'images': pages,
            'split_method': 'no_split',
            'num_pages': len(pages)
        }]
    
    def _group_pages_into_documents(
        self,
        pages: List[Image.Image],
        boundaries: List[int],
        source_filename: str,
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Group pages into documents based on detected boundaries."""
        documents = []
        
        for i, start_idx in enumerate(boundaries):
            # Determine end index
            if i < len(boundaries) - 1:
                end_idx = boundaries[i + 1]
            else:
                end_idx = len(pages)
            
            # Get pages for this document
            doc_pages = pages[start_idx:end_idx]
            page_indices = list(range(start_idx, end_idx))
            
            # Create document ID
            doc_id = f"{source_filename}_doc_{i+1:03d}"
            
            # Page range string
            if len(page_indices) == 1:
                page_range = f"{page_indices[0] + 1}"
            else:
                page_range = f"{page_indices[0] + 1}-{page_indices[-1] + 1}"
            
            documents.append({
                'source': source_filename,
                'pages': page_indices,
                'page_range': page_range,
                'document_id': doc_id,
                'images': doc_pages,
                'split_method': 'llm_intelligent',
                'num_pages': len(page_indices)
            })
            
            logger.info(f"  Document {i+1}: Pages {page_range} ({len(page_indices)} page{'s' if len(page_indices) > 1 else ''}) → {doc_id}")
        
        return documents
    
    def merge_document_pages(self, images: List[Image.Image]) -> Image.Image:
        """
        Merge multiple pages of a single document into one tall image.
        Used for processing multi-page documents as a single unit.
        """
        if len(images) == 1:
            return images[0]
        
        logger.info(f"  Merging {len(images)} pages of document into single image...")
        
        # Find max width
        target_width = max(img.width for img in images)
        
        # Resize all to same width and calculate total height
        resized_images = []
        total_height = 0
        
        for img in images:
            if img.width != target_width:
                ratio = target_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((target_width, new_height), Image.Resampling.LANCZOS)
            resized_images.append(img)
            total_height += img.height
        
        # Create merged image
        merged = Image.new('RGB', (target_width, total_height), 'white')
        
        y_offset = 0
        for img in resized_images:
            merged.paste(img, (0, y_offset))
            y_offset += img.height
        
        logger.info(f"  ✓ Merged into {target_width}x{total_height}px image")
        return merged
