"""
Classifier Agent Prompts - Auto-Generated from Registry
Enhanced with stricter instructions
"""

import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def _load_document_types():
    """
    Load document types from registry and group by category.
    
    Returns:
        dict: Document types grouped by category
    """
    try:
        registry_path = Path(__file__).parent.parent / 'config' / 'document_registry.yaml'
        
        with open(registry_path, 'r', encoding='utf-8') as f:
            registry = yaml.safe_load(f)
        
        # Group by category
        by_category = {}
        for doc in registry['documents']:
            category = doc['category']
            if category not in by_category:
                by_category[category] = []
            
            # Add type and description
            by_category[category].append({
                'type': doc['type'],
                'description': doc.get('description', '')
            })
        
        logger.debug(f"Loaded {len(registry['documents'])} document types from registry")
        return by_category
        
    except Exception as e:
        logger.error(f"Failed to load document types from registry: {e}")
        # Fallback to minimal set
        return {
            'financial': [{'type': 'other', 'description': 'Other financial document'}],
            'commercial': [{'type': 'other', 'description': 'Other commercial document'}],
            'transport': [{'type': 'other', 'description': 'Other transport document'}],
            'insurance': [{'type': 'other', 'description': 'Other insurance document'}],
            'unknown': [{'type': 'other', 'description': 'Unrecognized document'}]
        }


# Auto-load document types from registry
DOC_TYPES = _load_document_types()


def _build_document_list(category_name, doc_list):
    """Build formatted list of document types for a category."""
    lines = []
    for doc in doc_list:
        line = f"- {doc['type']}"
        if doc.get('description'):
            line += f": {doc['description']}"
        lines.append(line)
    return '\n'.join(lines)


# System prompt
SYSTEM_PROMPT = """You are an expert document classifier specializing in trade, financial, transport, and insurance documents.

CRITICAL RULES:
1. You MUST choose EXACTLY ONE type from the predefined list
2. Do NOT invent new type names - only use types from the list
3. Return ONLY valid JSON with no additional text
4. Use exact type names with underscores (e.g., insurance_policy, not insurance_document)"""


# Build the valid types list for the prompt
def _get_all_types():
    """Get flat list of all valid types."""
    all_types = []
    for category_docs in DOC_TYPES.values():
        for doc in category_docs:
            all_types.append(doc['type'])
    return all_types

ALL_VALID_TYPES = _get_all_types()


# Auto-generated classification prompt
CLASSIFICATION_PROMPT = f"""You are classifying a document into EXACTLY ONE of these predefined types.

VALID DOCUMENT TYPES (MUST USE EXACT NAMES):

FINANCIAL DOCUMENTS:
{_build_document_list('Financial', DOC_TYPES.get('financial', []))}

COMMERCIAL DOCUMENTS:
{_build_document_list('Commercial', DOC_TYPES.get('commercial', []))}

TRANSPORT DOCUMENTS:
{_build_document_list('Transport', DOC_TYPES.get('transport', []))}

INSURANCE DOCUMENTS:
{_build_document_list('Insurance', DOC_TYPES.get('insurance', []))}

OTHER:
{_build_document_list('Other', DOC_TYPES.get('unknown', []))}

IMPORTANT KEYWORDS TO IDENTIFY DOCUMENTS:
- Insurance Policy: Contains "policy number", "premium", "coverage amount", "insured party"
- Insurance Certificate: Contains "certificate of insurance", "certificate number"
- Commercial Invoice: Contains "invoice", "exporter", "importer", "goods", "total amount"
- Bill of Lading: Contains "bill of lading", "B/L", "shipper", "consignee", "vessel"
- Bill of Exchange: Contains "pay to the order of", "drawer", "drawee", "tenor"

DOCUMENT TEXT:
{{text}}

INSTRUCTIONS:
1. Read the document text carefully
2. Identify the document type based on keywords and structure
3. Choose EXACTLY ONE type from the list above (use the exact name with underscores)
4. If you see "insurance policy", return "insurance_policy" (not "insurance_document")
5. If you see "insurance certificate", return "insurance_certificate"
6. If unsure, use "other"

RETURN ONLY THIS JSON FORMAT (NO OTHER TEXT):
{{{{
  "document_type": "exact_type_from_list_with_underscores",
  "document_confidence": 0.95,
  "reasoning": "Found keywords: policy number, premium, coverage - this is an insurance_policy"
}}}}

CRITICAL: The document_type MUST be EXACTLY one of these {len(ALL_VALID_TYPES)} types:
{', '.join(ALL_VALID_TYPES[:20])}... (see full list above)

Do NOT invent new names like "insurance_document" - use the exact names from the list!"""


def get_classification_prompt(text: str) -> str:
    """
    Generate classification prompt with text sample.
    
    Args:
        text: Full OCR text
        
    Returns:
        str: Formatted prompt
    """
    # Limit text to first 3000 characters
    text_sample = text[:3000] if len(text) > 3000 else text
    return CLASSIFICATION_PROMPT.format(text=text_sample)


# Log loaded types on import
logger.info(f"Classifier prompt auto-generated with {len(ALL_VALID_TYPES)} document types")
logger.debug(f"Valid types: {', '.join(ALL_VALID_TYPES[:10])}...")
