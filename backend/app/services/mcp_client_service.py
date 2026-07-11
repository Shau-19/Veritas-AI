"""
MCP Client Service
Connects to external MCP (Model Context Protocol) servers as a consumer/client.
Lists available resources and reads their text content for fine-tuning dataset generation.

MCP Protocol Reference: https://spec.modelcontextprotocol.io
Transport: SSE (HTTP) — covers Notion, GitHub, Google Drive, Postgres, Slack integrations.
Stdio transport not supported in this version (requires subprocess management).

Design principles:
- SSE transport only (HTTP-based, works with all hosted MCP servers)
- Auth token passed as Bearer header
- Graceful error handling — lists what's available, skips unreadable resources
- Per-request timeout on all network calls
"""
import json
import logging
import asyncio
import httpx
from typing import Any

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 30.0  # seconds per MCP request


class MCPClientError(Exception):
    pass


async def _mcp_request(
    client: httpx.AsyncClient,
    server_url: str,
    method: str,
    params: dict | None = None,
) -> Any:
    """
    Make a JSON-RPC 2.0 request to an MCP server endpoint.
    MCP over HTTP uses POST with JSON-RPC body.
    """
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params or {},
    }

    try:
        response = await asyncio.wait_for(
            client.post(
                server_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            ),
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            raise MCPClientError(f"MCP server error: {data['error']}")

        return data.get("result", {})

    except asyncio.TimeoutError:
        raise MCPClientError(f"MCP server timed out at {server_url}")
    except httpx.HTTPError as e:
        raise MCPClientError(f"HTTP error connecting to MCP server: {e}")
    except json.JSONDecodeError:
        raise MCPClientError("MCP server returned non-JSON response")


def _build_client(auth_token: str | None) -> httpx.AsyncClient:
    """Build an httpx client with optional Bearer auth."""
    headers = {
        "User-Agent": "VeritasAI-MCP-Client/1.0",
        "Accept": "application/json",
    }
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    return httpx.AsyncClient(headers=headers, timeout=REQUEST_TIMEOUT)


async def list_mcp_resources(
    server_url: str,
    auth_token: str | None = None,
) -> list[dict]:
    """
    Connect to an MCP server and list all available resources.

    Args:
        server_url: Full URL to the MCP server endpoint (e.g. http://localhost:3001/mcp)
        auth_token: Optional Bearer token for authentication

    Returns:
        List of resource dicts: [{uri, name, description, mimeType}, ...]

    Raises:
        MCPClientError on connection or protocol failures
    """
    async with _build_client(auth_token) as client:
        # First: initialize the MCP session
        try:
            await _mcp_request(client, server_url, "initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {"roots": {}, "sampling": {}},
                "clientInfo": {"name": "VeritasAI", "version": "1.0"},
            })
        except MCPClientError as e:
            # Some servers skip explicit initialization — try listing anyway
            logger.warning(f"[MCP] Initialize handshake failed ({e}), proceeding anyway")

        # List resources
        result = await _mcp_request(client, server_url, "resources/list")
        resources = result.get("resources", [])

        logger.info(f"[MCP] Found {len(resources)} resources at {server_url}")
        return [
            {
                "uri": r.get("uri", ""),
                "name": r.get("name", r.get("uri", "Unknown")),
                "description": r.get("description", ""),
                "mimeType": r.get("mimeType", "text/plain"),
            }
            for r in resources
            if r.get("uri")
        ]


async def read_mcp_resources(
    server_url: str,
    resource_uris: list[str],
    auth_token: str | None = None,
) -> list[str]:
    """
    Read content from selected MCP resources and return as text chunks.

    Args:
        server_url: Full URL to the MCP server endpoint
        resource_uris: List of resource URI strings to read
        auth_token: Optional Bearer token

    Returns:
        List of text strings (one per successfully read resource)
    """
    if not resource_uris:
        return []

    async with _build_client(auth_token) as client:
        # Initialize session
        try:
            await _mcp_request(client, server_url, "initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "VeritasAI", "version": "1.0"},
            })
        except MCPClientError:
            pass  # Non-fatal

        texts: list[str] = []

        for uri in resource_uris:
            try:
                result = await _mcp_request(client, server_url, "resources/read", {
                    "uri": uri
                })

                contents = result.get("contents", [])
                for content_item in contents:
                    # Text content
                    if content_item.get("type") == "text" or "text" in content_item:
                        text = content_item.get("text", "")
                        if text and len(text.strip()) > 50:
                            texts.append(text.strip())
                    # Blob content (binary) — skip
                    elif content_item.get("type") == "blob":
                        logger.debug(f"[MCP] Skipping blob content at {uri}")

                logger.info(f"[MCP] Read {uri}: {len(texts)} text segments so far")

            except MCPClientError as e:
                logger.warning(f"[MCP] Could not read resource {uri}: {e}")
                continue

        return texts
