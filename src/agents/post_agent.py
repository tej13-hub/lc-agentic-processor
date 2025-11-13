"""
POST Agent - Handle document submission to API
"""

import logging
from typing import Dict, Any
from src.payload.payload_builder import PayloadBuilder
from src.mcp.mcp_client import MCPClient
from config.settings import settings

logger = logging.getLogger(__name__)


class PostAgent:
    """Agent that handles POST submission of documents."""
    
    def __init__(self, llm_client):
        """
        Initialize POST agent.
        
        Args:
            llm_client: LLM client for payload building
        """
        self.name = "POST AGENT"
        self.payload_builder = PayloadBuilder(llm_client)
        self.mcp_client = MCPClient()
        
        logger.info(f"{self.name} initialized")
    
    def submit_document(
        self,
        document_id: str,
        document_type: str,
        extracted_fields: dict
    ) -> Dict[str, Any]:
        """
        Submit document data to API.
        
        Args:
            document_id: Document identifier
            document_type: Type of document
            extracted_fields: Extracted field data
            
        Returns:
            Submission result
        """
        logger.info(f"\n{self.name}: Submitting document {document_id}")
        
        # Add document_id and type to extracted fields
        extracted_fields['document_id'] = document_id
        extracted_fields['document_type'] = document_type
        
        # Step 1: Build payload using LLM
        logger.info("  Step 1: Building payload...")
        build_result = self.payload_builder.build_payload(
            tool_name='setDocumentDetails',
            extracted_fields=extracted_fields
        )
        
        if not build_result['success']:
            logger.error(f"  ✗ Payload building failed: {build_result.get('error')}")
            return {
                'status': 'failed',
                'step': 'payload_building',
                'error': build_result
            }
        
        payload = build_result['payload']
        validation = build_result['validation']
        
        logger.info(f"  ✓ Payload built ({len(payload)} fields)")
        
        # Log which fields came from where
        logger.info(f"  Fields from document: {build_result['fields_from_doc']}")
        logger.info(f"  Fields from samples: {build_result['fields_from_sample']}")
        
        # Step 2: Validate payload
        if not validation['valid']:
            logger.error(f"  ✗ Validation failed:")
            for error in validation['errors']:
                logger.error(f"    - {error}")
            
            if settings.POST_VALIDATION:
                return {
                    'status': 'failed',
                    'step': 'validation',
                    'validation': validation,
                    'payload': payload
                }
        
        # Log warnings
        if validation['warnings']:
            logger.warning(f"  Validation warnings:")
            for warning in validation['warnings']:
                logger.warning(f"    - {warning}")
        
        # Step 3: Call MCP tool
        logger.info("  Step 2: Calling MCP tool...")
        response = self.mcp_client.call_tool(
            tool_name='setDocumentDetails',
            api_params=payload
        )
        
        # Step 4: Handle response
        if response['success']:
            logger.info(f"  ✓ SUCCESS: Document submitted")
            
            submission_id = None
            if isinstance(response['body'], dict):
                submission_id = response['body'].get('submission_id') or response['body'].get('id')
            
            if submission_id:
                logger.info(f"  Submission ID: {submission_id}")
            
            return {
                'status': 'success',
                'payload': payload,
                'validation': validation,
                'response': response,
                'submission_id': submission_id
            }
        
        else:
            logger.error(f"  ✗ FAILED: Status {response['status_code']}")
            
            if response['status_code'] == 400:
                logger.error(f"  Bad Request - Payload issue")
                logger.error(f"  API Error: {response.get('body')}")
            elif response['status_code'] == 500:
                logger.error(f"  Server Error - API issue")
            elif response['status_code'] == 408:
                logger.error(f"  Timeout - API took too long")
            else:
                logger.error(f"  Error: {response.get('error')}")
            
            return {
                'status': 'failed',
                'step': 'api_call',
                'payload': payload,
                'validation': validation,
                'response': response
            }
