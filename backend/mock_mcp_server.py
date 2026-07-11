from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

RESOURCES = [
    {
        "uri": "info://ai-history",
        "name": "History of Artificial Intelligence",
        "description": "A brief overview of AI progress, from Turing to transformers.",
        "mimeType": "text/plain",
        "text": "Artificial Intelligence began as a formal academic discipline in 1956 at a workshop on the campus of Dartmouth College. John McCarthy coined the term there. Early pioneers like Marvin Minsky and Alan Turing laid the mathematical and cognitive foundations. Over the decades, AI went through several cycles of optimism followed by periods of disappointment and funding cuts, known as AI winters. In the 2010s, deep learning and neural networks revolutionized computer vision and natural language processing. In the 2020s, Large Language Models based on the transformer architecture became the state of the art, demonstrating advanced text generation and reasoning capabilities."
    },
    {
        "uri": "info://india-geography",
        "name": "Geography of India",
        "description": "A summary of India's geographic regions, landscapes, and borders.",
        "mimeType": "text/plain",
        "text": "India is located in South Asia, bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast. It shares land borders with Pakistan to the west; China, Nepal, and Bhutan to the north; and Bangladesh and Myanmar to the east. The country features diverse landscapes, including the snow-capped Himalayan mountain range in the north, the fertile Indo-Gangetic plains, the Thar Desert in the west, and the Deccan Plateau in the south. Major rivers include the Ganges, Brahmaputra, and Indus, which support agricultural lands and major populations."
    }
]

@app.post("/mcp")
async def mcp_endpoint(request: Request):
    body = await request.json()
    method = body.get("method")
    params = body.get("params", {})
    rpc_id = body.get("id", 1)
    
    print(f"[Mock-MCP-Server] Received request: method={method}, params={params}")
    
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "serverInfo": {"name": "Mock-MCP-Server", "version": "1.0"}
            }
        }
    elif method == "resources/list":
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "resources": [
                    {
                        "uri": r["uri"],
                        "name": r["name"],
                        "description": r["description"],
                        "mimeType": r["mimeType"]
                    } for r in RESOURCES
                ]
            }
        }
    elif method == "resources/read":
        uri = params.get("uri")
        resource = next((r for r in RESOURCES if r["uri"] == uri), None)
        if not resource:
            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {
                    "code": -32602,
                    "message": f"Resource not found: {uri}"
                }
            }
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "contents": [
                    {
                        "type": "text",
                        "text": resource["text"]
                    }
                ]
            }
        }
    else:
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}"
            }
        }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
