import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.mcp_client_service import list_mcp_resources, read_mcp_resources, MCPClientError


async def test_mcp_list_resources():
    server_url = "http://127.0.0.1:8001/mcp"
    try:
        resources = await list_mcp_resources(server_url)
        print(f"[OK] Successfully listed resources from MCP server. Found {len(resources)} items:")
        for r in resources:
            print(f"  - {r['name']} ({r['uri']}): {r['description']}")
            
        assert len(resources) == 2
        assert resources[0]["uri"] == "info://ai-history"
        assert resources[1]["uri"] == "info://india-geography"
        print("[OK] Resource list structure and content match expected values.")
    except Exception as e:
        print(f"[ERROR] Failed to list resources: {e}")
        raise e


async def test_mcp_read_resources():
    server_url = "http://127.0.0.1:8001/mcp"
    uris = ["info://ai-history"]
    try:
        texts = await read_mcp_resources(server_url, uris)
        print(f"[OK] Successfully read resources from MCP server. Found {len(texts)} text segments:")
        for t in texts:
            print(f"  - Chunk size: {len(t)} chars: \"{t[:60]}...\"")
            
        assert len(texts) == 1
        assert "Artificial Intelligence" in texts[0]
        assert "1956" in texts[0]
        print("[OK] Resource content read matches expected mock values.")
    except Exception as e:
        print(f"[ERROR] Failed to read resources: {e}")
        raise e


if __name__ == "__main__":
    print("=== Running MCP Integration Tests against Mock MCP Server ===")
    asyncio.run(test_mcp_list_resources())
    asyncio.run(test_mcp_read_resources())
    print("=== All MCP Integration Tests Passed Successfully! ===")
