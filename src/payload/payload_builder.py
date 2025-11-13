"""
Payload Builder - With $ref Resolution and Sample Loading
"""

import logging
import httpx
from typing import Dict, Any
from prompts.payload_prompts import build_payload_prompt
from src.payload.payload_parser import PayloadParser
from src.payload.payload_validator import PayloadValidator
from src.payload.schema_resolver import SchemaResolver
from src.payload.sample_loader import SampleLoader
from config.settings import settings

logger = logging.getLogger(__name__)


class PayloadBuilder:
    """Build API payload using LLM with dynamic schema fetching and resolution."""
    
    def __init__(self, llm_client, mcp_server_url: str = None):
        """
        Initialize payload builder.
        
        Args:
            llm_client: LLM client for generation
            mcp_server_url: MCP server URL (defaults to settings)
        """
        self.llm = llm_client
        self.parser = PayloadParser()
        self.mcp_url = mcp_server_url or settings.MCP_SERVER_URL
        
        # Initialize schema resolver and sample loader
        self.schema_resolver = SchemaResolver()
        self.sample_loader = SampleLoader()
        
        logger.info(f"PayloadBuilder initialized (MCP: {self.mcp_url})")
    
    def fetch_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """
        Fetch tool schema from MCP server.
        
        Args:
            tool_name: Name of tool (e.g., "setDocumentDetails")
            
        Returns:
            Tool configuration with inputSchema
            
        Raises:
            ValueError: If tool not found or API fails
        """
        endpoint = f"{self.mcp_url}/tools/list"
        
        logger.info(f"Fetching schema from MCP: {endpoint}")
        
        try:
            response = httpx.get(endpoint, timeout=10.0)
            response.raise_for_status()
            
            data = response.json()
            tools = data.get('tools', [])
            
            # Find the requested tool
            for tool in tools:
                if tool.get('name') == tool_name:
                    logger.info(f"✓ Found schema for tool: {tool_name}")
                    return tool
            
            # Tool not found
            raise ValueError(f"Tool '{tool_name}' not found in MCP server response")
        
        except httpx.TimeoutException:
            logger.error(f"Timeout fetching schema from {endpoint}")
            raise ValueError(f"MCP server timeout at {endpoint}")
        
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching schema: {e.response.status_code}")
            raise ValueError(f"MCP server returned error: {e.response.status_code}")
        
        except Exception as e:
            logger.error(f"Failed to fetch schema: {e}")
            raise ValueError(f"Could not fetch schema from MCP server: {str(e)}")
    
    def build_payload(
        self,
        tool_name: str,
        extracted_fields: dict
    ) -> Dict[str, Any]:
        """
        Build payload for MCP tool - ALL fields must be populated.
        
        Args:
            tool_name: Name of MCP tool
            extracted_fields: Fields extracted from document
            
        Returns:
            Result dict with payload and metadata
        """
        logger.info(f"Building payload for tool: {tool_name}")
        logger.info("="*70)
        
        # Step 1: Fetch tool configuration from MCP server
        logger.info("Step 1: Fetching schema from MCP server...")
        try:
            tool_config = self.fetch_tool_schema(tool_name)
        except ValueError as e:
            logger.error(f"✗ Schema fetch failed: {e}")
            return {
                'success': False,
                'error': 'schema_fetch_failed',
                'message': str(e)
            }
        
        input_schema = tool_config.get('inputSchema', {})
        
        if not input_schema:
            logger.error(f"✗ Tool {tool_name} has no inputSchema")
            return {
                'success': False,
                'error': 'no_schema',
                'message': 'Tool has no inputSchema defined'
            }
        
        # Step 2: Resolve $refs in schema
        logger.info("Step 2: Resolving $ref references...")
        resolved_schema = self.schema_resolver.resolve(input_schema)
        
        # Get all expected fields
        all_fields = self.schema_resolver.get_all_field_paths(resolved_schema)
        logger.info(f"✓ Schema resolved: {len(all_fields)} fields expected")
        
        # Step 3: Load sample payload from JSON file
        logger.info("Step 3: Loading sample payload from JSON...")
        sample_payload = self.sample_loader.load_sample(tool_name)
        
        if not sample_payload:
            logger.error(f"✗ No sample file found")
            return {
                'success': False,
                'error': 'no_sample',
                'message': f'Sample file not found: config/samples/{tool_name}.json'
            }
        
        # Step 4: Build LLM prompt
        logger.info("Step 4: Building LLM prompt...")
        prompt = build_payload_prompt(
            tool_name=tool_name,
            resolved_schema=resolved_schema,
            extracted_fields=extracted_fields,
            sample_payload=sample_payload
        )
        
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        # Step 5: Call LLM
        logger.info("Step 5: Calling LLM to fill payload...")
        try:
            llm_response = self.llm.generate(
                prompt=prompt,
                system_prompt="You are a precise data mapping assistant. Populate ALL fields. Return only valid JSON.",
                timeout=60
            )
            
            logger.debug(f"LLM response length: {len(llm_response)} characters")
            
        except Exception as e:
            logger.error(f"✗ LLM generation failed: {e}")
            return {
                'success': False,
                'error': 'llm_generation_failed',
                'message': str(e)
            }
        
        # Step 6: Parse LLM response
        logger.info("Step 6: Parsing LLM response...")
        payload = self.parser.parse(llm_response)
        
        if not payload:
            logger.error("✗ Failed to parse payload from LLM response")
            return {
                'success': False,
                'error': 'parse_failed',
                'llm_response': llm_response[:500]
            }
        
        logger.info(f"✓ Payload parsed: {len(payload)} top-level fields")
        
        # Step 7: Validate payload
        logger.info("Step 7: Validating payload...")
        validator = PayloadValidator(resolved_schema)
        validation_result = validator.validate(
            payload=payload,
            extracted_fields=extracted_fields,
            sample_payload=sample_payload
        )
        
        logger.info("="*70)
        
        return {
            'success': True,
            'payload': payload,
            'validation': validation_result,
            'llm_response': llm_response,
            'fields_from_doc': validation_result.get('fields_from_doc', []),
            'fields_from_sample': validation_result.get('fields_from_sample', []),
            'schema_source': 'mcp_server',
            'total_fields_expected': len(all_fields),
            'resolved_schema': resolved_schema
        }
