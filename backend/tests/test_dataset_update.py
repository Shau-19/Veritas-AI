import sys
import os
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.dataset import Dataset
from app.api.routes.datasets import update_dataset, UpdateDatasetRequest, DatasetResponse
from app.services.scraper_service import scrape_url


# 1. Test Dataset PUT Endpoint Row & Name Updates
async def test_update_dataset_endpoint():
    dataset_id = "test-dataset-123"
    user_id = "test-user-456"
    
    # Mock dataset in DB
    mock_dataset = Dataset(
        id=dataset_id,
        user_id=user_id,
        name="Original Name",
        description="Original Description",
        row_count=2,
        file_size_bytes=100,
        file_format="jsonl",
        quality_score=85.0,
        audit_results={
            "rows": [
                {"input": "in 1", "output": "out 1"},
                {"input": "in 2", "output": "out 2"}
            ]
        },
        status="ready"
    )
    
    # Mock db session execute returns
    mock_session = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()
    
    class MockResult:
        def __init__(self, scalar_val):
            self._val = scalar_val
        def scalar_one_or_none(self):
            return self._val
            
    mock_session.execute.return_value = MockResult(mock_dataset)
    
    # Request body with edits
    edit_request = UpdateDatasetRequest(
        name="New Updated Name",
        rows=[
            {"input": "in 1 edited", "output": "out 1 edited"},
            {"input": "in 2 edited", "output": "out 2 edited"},
            {"input": "in 3 new", "output": "out 3 new"}
        ]
    )
    
    # Call route function
    response = await update_dataset(
        dataset_id=dataset_id,
        body=edit_request,
        user_id=user_id,
        db=mock_session
    )
    
    # Asserts
    assert response.name == "New Updated Name"
    assert response.row_count == 3
    assert len(response.audit_results["rows"]) == 3
    assert response.audit_results["rows"][0]["input"] == "in 1 edited"
    assert response.audit_results["rows"][2]["input"] == "in 3 new"
    
    # Check that file size was recalculated
    expected_size = len(json.dumps(edit_request.rows).encode("utf-8"))
    assert response.file_size_bytes == expected_size
    
    # Verify database calls
    mock_session.execute.assert_called_once()
    mock_session.commit.assert_called_once()
    mock_session.refresh.assert_called_once()
    
    print("[OK] test_update_dataset_endpoint passed")


# 2. Test Multi-Link Scraper with List URLs
async def test_scrape_multi_url_yields():
    urls = [
        "https://example.com/page1",
        "https://example.com/page2"
    ]
    
    # Mock HTTPX AsyncClient requests
    class MockResponse:
        def __init__(self, text, status_code=200):
            self.text = text
            self.status_code = status_code
            self.headers = {"content-type": "text/html"}

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock()
    
    # Mock response for sitemaps (return None/404) and normal pages (return HTML text)
    async def mock_get(url, **kwargs):
        if "sitemap" in url:
            return MockResponse("", 404)
        elif "page1" in url:
            return MockResponse("<html><body><main><p>This is the first page of content for testing multi url support. It needs to contain a sufficient amount of text to pass the minimum chunk size validation which is eighty characters in total.</p></main></body></html>")
        elif "page2" in url:
            return MockResponse("<html><body><main><p>This is the second page of content containing different details for testing. It also needs to contain a sufficient amount of text to pass the minimum chunk size validation which is eighty characters in total.</p></main></body></html>")
        return MockResponse("", 404)
        
    mock_client.get = mock_get
    
    with patch("httpx.AsyncClient", return_value=mock_client), \
         patch("asyncio.sleep", AsyncMock()):
         
        events = []
        async for event in scrape_url(urls, max_pages=5):
            events.append(event)
            
        # Verify page_scraped events
        scraped_events = [e for e in events if e["event"] == "page_scraped"]
        assert len(scraped_events) == 2
        assert scraped_events[0]["url"] == "https://example.com/page1"
        assert scraped_events[1]["url"] == "https://example.com/page2"
        
        # Verify complete event has compiled all chunks
        complete_event = next(e for e in events if e["event"] == "complete")
        assert complete_event["pages_scraped"] == 2
        assert len(complete_event["chunks"]) >= 2
        
        # Check text was extracted correctly
        combined_text = "".join(complete_event["chunks"])
        assert "first page of content" in combined_text
        assert "second page of content" in combined_text
        
        print("[OK] test_scrape_multi_url_yields passed")


if __name__ == "__main__":
    print("=== Running Dataset Update & Multi-Link Scraper Unit Tests ===")
    asyncio.run(test_update_dataset_endpoint())
    asyncio.run(test_scrape_multi_url_yields())
    print("=== All tests passed successfully! ===")
