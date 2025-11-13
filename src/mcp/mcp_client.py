"""
MCP Client - Call MCP tools via HTTP
"""

import logging
import httpx
from typing import Dict, Any
from config.settings import settings

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for calling MCP tools."""
    
    def __init__(self, server_url: str = None):
        """
        Initialize MCP client.
        
        Args:
            server_url: MCP server URL (defaults to settings)
        """
        self.server_url = server_url or settings.MCP_SERVER_URL
        self.timeout = settings.MCP_TIMEOUT
        
        logger.info(f"MCPClient initialized: {self.server_url}")
    
    def call_tool(self, tool_name: str, api_params: dict) -> Dict[str, Any]:
        """
        Call MCP tool with parameters.
        
        Args:
            tool_name: Name of tool (e.g., "setDocumentDetails")
            api_params: Payload to send to tool
            
        Returns:
            Response dict with status, body, success flag
        """
        endpoint = f"{self.server_url}/tools/{tool_name}"
        
        logger.info(f"Calling MCP tool: {tool_name}")
        logger.debug(f"Endpoint: {endpoint}")
        logger.debug(f"Payload: {api_params}")
        
        try:
            response = httpx.post(
                endpoint,
                json=api_params,
                timeout=self.timeout
            )
            
            status_code = response.status_code
            success = status_code in [200, 201]
            
            # Try to parse JSON response
            try:
                body = response.json()
            except:
                body = response.text
            
            logger.info(f"Response: {status_code} ({'success' if success else 'failed'})")
            
            return {
                'status_code': status_code,
                'body': body,
                'success': success,
                'error': None
            }
        
        except httpx.TimeoutException:
            logger.error(f"Request timeout after {self.timeout}s")
            return {
                'status_code': 408,
                'body': None,
                'success': False,
                'error': 'timeout'
            }
        
        except httpx.ConnectError as e:
            logger.error(f"Connection failed: {e}")
            return {
                'status_code': 503,
                'body': None,
                'success': False,
                'error': f'connection_error: {str(e)}'
            }
        
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {
                'status_code': 500,
                'body': None,
                'success': False,
                'error': str(e)
            }
