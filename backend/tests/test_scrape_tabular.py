import sys
import os
import json
import pytest
from unittest.mock import AsyncMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.restructurer_service import extract_tabular_from_chunks


@pytest.mark.asyncio
async def test_extract_tabular_from_chunks():
    chunks = [
        "Company Alpha was founded in 2015 by Alice. It has 50 employees in total and operates globally.",
        "Company Beta was founded in 2018 by Bob. It has 120 employees in total and operates globally."
    ]
    columns = [
        {"name": "company_name", "type": "string"},
        {"name": "founded_year", "type": "int"},
        {"name": "employee_count", "type": "int"}
    ]

    mock_llm_responses = [
        # Response for chunk 0
        {
            "rows": [
                {"company_name": "Company Alpha", "founded_year": 2015, "employee_count": 50}
            ]
        },
        # Response for chunk 1
        {
            "rows": [
                {"company_name": "Company Beta", "founded_year": 2018, "employee_count": 120}
            ]
        }
    ]

    # Create a mock OpenAI client
    mock_client = AsyncMock()
    
    class MockChoice:
        class MockMessage:
            def __init__(self, content):
                self.content = content
        def __init__(self, content):
            self.message = MockChoice.MockMessage(content)

    # Configure mock completions generator
    completion_calls = []
    async def mock_create(**kwargs):
        idx = len(completion_calls)
        completion_calls.append(kwargs)
        # Return mock JSON response matching each chunk
        response_content = json.dumps(mock_llm_responses[idx])
        mock_resp = AsyncMock()
        mock_resp.choices = [MockChoice(response_content)]
        return mock_resp

    mock_client.chat.completions.create = mock_create
    mock_client._client.cookies.clear = lambda: None

    with patch("app.services.llm_client.get_llm_client", return_value=(mock_client, "gpt-4o")):
        progress_calls = []
        async def on_progress(chunk_idx, completed_chunks, num_rows, total_rows_so_far, elapsed):
            progress_calls.append((chunk_idx, completed_chunks, num_rows, total_rows_so_far))

        rows = await extract_tabular_from_chunks(
            chunks=chunks,
            columns=columns,
            on_progress=on_progress
        )

        assert len(rows) == 2
        assert rows[0]["company_name"] == "Company Alpha"
        assert rows[0]["founded_year"] == 2015
        assert rows[0]["employee_count"] == 50
        assert rows[1]["company_name"] == "Company Beta"
        assert rows[1]["founded_year"] == 2018
        assert rows[1]["employee_count"] == 120

        # Assert progress callback was invoked for both chunks
        assert len(progress_calls) == 2
        assert progress_calls[0][1] == 1
        assert progress_calls[1][1] == 2
        print("\n[OK] extract_tabular_from_chunks passed successfully!")
