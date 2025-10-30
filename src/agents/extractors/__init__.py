# """
# Document Extractors Package
# """

# from src.agents.extractors.base_extractor import BaseExtractor
# from src.agents.extractors.bill_of_exchange_extractor import BillOfExchangeExtractor
# from src.agents.extractors.draft_extractor import DraftExtractor
# from src.agents.extractors.cheque_extractor import ChequeExtractor
# from src.agents.extractors.demand_draft_extractor import DemandDraftExtractor
# from src.agents.extractors.purchase_order_extractor import PurchaseOrderExtractor
# from src.agents.extractors.proforma_invoice_extractor import ProformaInvoiceExtractor
# from src.agents.extractors.commercial_invoice_extractor import CommercialInvoiceExtractor

# __all__ = [
#     'BaseExtractor',
#     'BillOfExchangeExtractor',
#     'DraftExtractor',
#     'ChequeExtractor',
#     'DemandDraftExtractor',
#     'PurchaseOrderExtractor',
#     'ProformaInvoiceExtractor',
#     'CommercialInvoiceExtractor'
# ]

"""
Document Extractors Package
Now uses DynamicExtractor for all document types!
"""

from src.agents.extractors.dynamic_extractor import DynamicExtractor

__all__ = ['DynamicExtractor']
