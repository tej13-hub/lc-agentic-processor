"""
Classifier Agent Prompts - Enhanced with better examples
"""

SYSTEM_PROMPT = """You are an expert document classifier specializing in trade, financial, transport, and insurance documents.

Your task: Analyze the document carefully and classify it into EXACTLY ONE of the predefined categories.

CRITICAL: Be precise. Look for specific keywords and document structure patterns."""


CLASSIFICATION_PROMPT = """Analyze this document text and classify it into EXACTLY ONE category from the list below.

IMPORTANT CLASSIFICATION GUIDELINES:
- Insurance documents contain: policy number, premium, coverage, insured party, beneficiary
- Commercial invoices contain: invoice number, exporter, importer, goods description, prices
- Bills of exchange contain: drawer, drawee, amount payable, tenor, maturity date
- Certificates contain: certificate number, issuing authority, validity dates

DOCUMENT TYPES:

FINANCIAL DOCUMENTS:
- bill_of_exchange: Payment instrument with drawer/drawee
- draft: Similar to bill of exchange
- cheque: Bank cheque with cheque number
- demand_draft: Bank DD with reference number

COMMERCIAL DOCUMENTS:
- purchase_order: PO number, vendor, buyer, items to purchase
- proforma_invoice: PI number, quotation before shipment
- commercial_invoice: Invoice for international trade with exporter/importer
- tax_invoice: Tax invoice with GST/VAT details
- debit_note: Debit adjustment document
- credit_note: Credit adjustment document
- cash_memo: Cash sale receipt
- mt103_swift: SWIFT payment message MT103
- mt202_swift: SWIFT payment message MT202
- packing_list: List of packages and contents
- weight_list: Weight details of shipment
- contract: Legal agreement/contract
- certificate_of_origin: Origin certificate for goods
- gsp_certificate: GSP certificate for preferential tariffs
- inspection_certificate: Inspection report/certificate
- fumigation_certificate: Fumigation treatment certificate
- warranty_certificate: Product warranty certificate
- certificate_of_analysis: Analysis report certificate
- certificate_of_quality: Quality certificate
- phytosanitary_certificate: Plant health certificate
- iec_code_certificate: Import Export Code certificate
- shipping_bill: Customs shipping bill for export
- bill_of_entry: Customs bill for import
- export_import_license: Export or import license
- sdf_form: Statutory Declaration Form
- softex: Software export form
- high_seas_agreement: High seas sale agreement
- shipping_guarantee: Shipping bank guarantee
- sales_agreement: Sales contract/agreement
- promissory_note: Promissory note for payment
- gr_waiver: General Remittance Waiver

TRANSPORT DOCUMENTS:
- bill_of_lading: Shipping document from carrier
- airway_bill: Air cargo document
- multimodal_transport_document: Combined transport document
- lorry_receipt: Truck/lorry transport receipt
- cmr: Road transport document (CMR convention)
- delivery_note: Delivery challan/note
- courier_receipt: Courier shipment receipt
- postal_receipt: Postal service receipt
- seaway_bill: Sea waybill

INSURANCE DOCUMENTS:
- insurance_policy: Insurance policy with coverage details, premium, policy number
- insurance_certificate: Certificate of insurance
- consignment_advice_insurance: Advice for insurance of consignment
- cover_note: Insurance cover note
- marine_insurance: Marine cargo insurance
- cargo_insurance: Cargo insurance policy
- customs_bond: Customs surety bond

OTHER:
- other: Unrecognized document type

DOCUMENT TEXT:
{text}

CLASSIFICATION STEPS:
1. Read the document carefully
2. Identify key fields (policy number vs invoice number vs certificate number)
3. Look for specific terminology (premium, coverage = insurance; exporter, goods = commercial)
4. Choose the MOST specific matching type

RETURN FORMAT (ONLY JSON, NO OTHER TEXT):
{{
  "document_type": "exact_type_from_above",
  "document_confidence": 0.95,
  "reasoning": "Found policy number X, premium amount Y, insured party Z - typical insurance policy structure"
}}

CRITICAL RULES:
- Return ONLY the JSON object
- No markdown, no code blocks, pure JSON
- document_type MUST exactly match one from the list (use underscores)
- document_confidence: 0.0 to 1.0
- reasoning: Explain which specific keywords/patterns you found"""


def get_classification_prompt(text: str) -> str:
    """Generate classification prompt with text sample."""
    # Limit text to first 3000 characters
    text_sample = text[:3000] if len(text) > 3000 else text
    return CLASSIFICATION_PROMPT.format(text=text_sample)
