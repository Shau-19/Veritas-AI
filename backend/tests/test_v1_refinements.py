import sys
import os
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.dataset_generator import generate_dataset_stream, verify_semantic_coverage, robust_json_loads
from app.services.structured_generator import generate_structured_stream, infer_conditional_dependencies
from app.services.repair_engine import detect_weak_structured_rows, repair_structured_dataset

# Mock local embedder class for offline unit testing
class MockLocalEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        # Return unique mock vectors to prevent duplicate detection in audits
        res = []
        for idx, text in enumerate(texts):
            vec = [0.0] * 384
            # Use hash of text to make it distinct
            val = float(hash(text) % 10000) / 10000.0
            vec[idx % 384] = val if val > 0 else 0.5
            res.append(vec)
        return res

# 1. Test Auto-Inferred Conditional Dependencies
def test_conditional_dependencies_inference():
    columns = [
        {"name": "income", "type": "float"},
        {"name": "loan_amount", "type": "float"},
        {"name": "credit_score", "type": "int"},
        {"name": "default_risk", "type": "category"}
    ]
    rules, confidence = infer_conditional_dependencies(columns)
    assert len(rules) >= 3, f"Expected at least 3 rules, got {len(rules)}"
    assert confidence > 0.75, f"Expected confidence > 0.75, got {confidence}"
    print("[OK] test_conditional_dependencies_inference passed")

# 2. Test Weak Structured Row Detection
def test_detect_weak_structured_rows():
    columns = [
        {"name": "age", "type": "int"},
        {"name": "job_title", "type": "string"},
        {"name": "target", "type": "category"}
    ]
    rows = [
        {"age": 10, "job_title": "Senior Engineer", "target": "yes"}, # Logic violation
        {"age": 30, "job_title": "Dentist", "target": ""},            # Empty target
        {"age": 40, "job_title": "Teacher", "target": "no"},           # Good row
        {"age": 40, "job_title": "Teacher", "target": "no"}            # Duplicate row
    ]
    
    weak = detect_weak_structured_rows(rows, columns)
    assert len(weak) == 3, f"Expected 3 weak rows, got {len(weak)}"
    indices = [w["index"] for w in weak]
    assert 0 in indices, "Expected row 0 to be flagged (age vs senior job)"
    assert 1 in indices, "Expected row 1 to be flagged (missing target)"
    assert 3 in indices, "Expected row 3 to be flagged (duplicate)"
    print("[OK] test_detect_weak_structured_rows passed")

# 3. Test Semantic Coverage Verification
def test_verify_semantic_coverage():
    rows = [
        {"input": "a b c d", "output": "Substantive output details go here.", "subdomain": "sub1"},
        {"input": "a b c d e", "output": "Substantive output details go here.", "subdomain": "sub1"},
        {"input": "a b c d f", "output": "Substantive output details go here.", "subdomain": "sub1"},
        {"input": "x y z w", "output": "Substantive output details go here.", "subdomain": "sub2"},
    ]
    coverage_plan = {"sub1": 3, "sub2": 3}
    res = verify_semantic_coverage(rows, coverage_plan)
    assert not res["is_coverage_sufficient"], "Coverage should not be sufficient"
    assert len(res["underrepresented_subdomains"]) == 1, "sub2 should be underrepresented"
    assert res["underrepresented_subdomains"][0]["subdomain"] == "sub2"
    print("[OK] test_verify_semantic_coverage passed")

# Mock OpenAI client classes
class MockChatCompletion:
    def __init__(self, content):
        self.choices = [MagicMock(message=MagicMock(content=content))]

class MockEmbeddingData:
    def __init__(self, embedding):
        self.embedding = embedding

class MockEmbeddingResponse:
    def __init__(self, embeddings):
        self.data = [MockEmbeddingData(emb) for emb in embeddings]

# 4. Test Text Generator Dynamic Entropy Recovery & Provenance Mocked
async def test_text_generator_refinements():
    # Repetitive starters to mock the generator output
    mock_rows = [
        {"input": f"Distinct input query {i}", "output": f"Output {i} sequence is a unique response details."}
        for i in range(12)
    ]
    mock_response_json = json.dumps(mock_rows)
    
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=MockChatCompletion(mock_response_json))
    mock_client.embeddings.create = AsyncMock(return_value=MockEmbeddingResponse([[0.1]*384 for _ in range(12)]))
    mock_client.close = AsyncMock()

    mock_embedder = MockLocalEmbedder()

    with patch('app.services.dataset_generator.get_llm_client', return_value=(mock_client, "gpt-4o-mini")), \
         patch('app.services.dataset_generator.get_openai_client', return_value=mock_client), \
         patch('app.services.llm_client.get_embedding_client', return_value=mock_embedder), \
         patch('app.services.dataset_generator.profile_dataset_request', return_value={"factuality_risk": "low", "style_entropy_needed": "medium", "edge_case_ratio": 0.1, "primary_personas": ["expert analyst"], "dataset_name": "Test", "task_type": "chat"}), \
         patch('app.services.dataset_generator.plan_dataset_coverage', return_value={"Core Concepts": 10}):
        events = []
        async for event in generate_dataset_stream(
            topic="test",
            task_type="chat",
            style="technical",
            total_rows=10
        ):
            events.append(event)
            
        complete_event = next(e for e in events if e["event"] == "complete")
        assert "rows" in complete_event
        rows = complete_event["rows"]
        assert len(rows) > 0
        # Check provenance tracking exists
        for r in rows:
            assert "_provenance" in r
            assert r["_provenance"]["batch"] in (1, 2, "repair_loop")
            assert "temperature" in r["_provenance"]
            
        # Check filter tracking
        batch_done_events = [e for e in events if e["event"] == "batch_done"]
        assert len(batch_done_events) > 0
        for e in batch_done_events:
            assert "filter_reasons" in e
            assert "empty_io" in e["filter_reasons"]
            assert "too_short" in e["filter_reasons"]
            assert "semantic_duplicate" in e["filter_reasons"]
            
        print("[OK] test_text_generator_refinements passed")

# 5. Test Structured Stream SSE Complete Payload & Provenance
async def test_structured_stream_refinements():
    mock_rows = [
        {"income": 50000 + i * 1000, "credit_score": 600 + i * 5, "default_risk": "no"}
        for i in range(15)
    ]
    mock_response_json = json.dumps({"rows": mock_rows})
    
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=MockChatCompletion(mock_response_json))
    mock_client.close = AsyncMock()

    with patch('app.services.structured_generator.get_llm_client', return_value=(mock_client, "gpt-4o-mini")):
        events = []
        async for event in generate_structured_stream(
            description="house prices",
            columns=[
                {"name": "income", "type": "float"},
                {"name": "credit_score", "type": "int"},
                {"name": "default_risk", "type": "category"}
            ],
            total_rows=15,
            missingness_rate=0.10,
            label_noise_rate=0.05
        ):
            events.append(event)
            
        complete_event = next(e for e in events if e["event"] == "complete")
        assert "rows" in complete_event
        rows = complete_event["rows"]
        assert len(rows) > 0
        for r in rows:
            assert "_provenance" in r
            assert r["_provenance"]["batch"] == 1
            assert r["_provenance"]["missingness_applied"] is True
            assert r["_provenance"]["label_noise_applied"] is True

        assert "audit" in complete_event
        print("[OK] test_structured_stream_refinements passed")

def test_robust_json_loads():
    # 1. Test markdown wrapping
    tc1 = '```json\n[{"input": "a", "output": "b"}]\n```'
    assert robust_json_loads(tc1) == [{"input": "a", "output": "b"}]
    
    # 2. Test raw newlines (strict=False)
    tc2 = '[\n  {\n    "input": "Write code",\n    "output": "def foo():\n    print(\'hello\')"\n  }\n]'
    res2 = robust_json_loads(tc2)
    assert len(res2) == 1
    assert res2[0]["input"] == "Write code"
    
    # 3. Test invalid escape sequence cleaning
    tc3 = '[\n  {\n    "input": "Regex",\n    "output": "Match \\x or \\y"\n  }\n]'
    res3 = robust_json_loads(tc3)
    assert len(res3) == 1
    assert "Match \\x or \\y" in res3[0]["output"]

    # 4. Test brace balancing & regex fallback for unescaped double quotes inside value
    tc4 = '''[
      {
        "input": "Write python print",
        "output": "print("Hello, World!")"
      }
    ]'''
    res4 = robust_json_loads(tc4)
    assert len(res4) == 1
    assert res4[0]["output"] == 'print("Hello, World!")'
    
    print("[OK] test_robust_json_loads passed")

# 6. Test Concurrent Lock Contention
async def test_concurrent_lock_contention():
    all_rows = []
    lock = asyncio.Lock()
    
    async def append_worker(rows_to_add):
        async with lock:
            await asyncio.sleep(0.005)
            all_rows.extend(rows_to_add)
            
    workers = [append_worker([{"id": f"w{i}_r{j}"} for j in range(5)]) for i in range(10)]
    await asyncio.gather(*workers)
    
    assert len(all_rows) == 50
    ids = [r["id"] for r in all_rows]
    assert len(set(ids)) == 50
    print("[OK] test_concurrent_lock_contention passed")

# 7. Test DeepSeek fallback on consecutive failures
async def test_deepseek_fallback_on_consecutive_failures():
    call_count = 0
    
    async def mock_deepseek_completions(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            raise Exception("DeepSeek Rate Limit/Outage Exception")
        # Returns a mock chat completion
        return MockChatCompletion(json.dumps([{"input": f"fallback input {call_count}", "output": f"fallback output {call_count} is long enough to pass filters successfully."}]))
        
    mock_deepseek_client = MagicMock()
    mock_deepseek_client.chat.completions.create = AsyncMock(side_effect=mock_deepseek_completions)
    mock_deepseek_client.close = AsyncMock()
    
    mock_openai_response = MockChatCompletion(json.dumps([{"input": "openai input", "output": "openai output sequence is long enough to pass filters successfully."}]))
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
    mock_openai_client.close = AsyncMock()
    
    mock_embedder = MockLocalEmbedder()
    
    with patch('app.services.dataset_generator.get_llm_client', return_value=(mock_deepseek_client, "deepseek-chat")), \
         patch('app.services.dataset_generator.get_openai_client', return_value=mock_openai_client), \
         patch('app.services.llm_client.get_embedding_client', return_value=mock_embedder), \
         patch('app.services.dataset_generator.profile_dataset_request', return_value={"factuality_risk": "low", "style_entropy_needed": "medium", "edge_case_ratio": 0.1, "primary_personas": ["expert analyst"], "dataset_name": "Test", "task_type": "chat"}), \
         patch('app.services.dataset_generator.plan_dataset_coverage', return_value={"Core Concepts": 5}):
        
        events = []
        async for event in generate_dataset_stream(
            topic="test fallback topic",
            task_type="chat",
            style="formal",
            total_rows=5
        ):
            events.append(event)
            
        complete_event = next(e for e in events if e["event"] == "complete")
        rows = complete_event["rows"]
        assert len(rows) > 0
        
        # Check filter tracking
        batch_done_events = [e for e in events if e["event"] == "batch_done"]
        assert len(batch_done_events) > 0
        for e in batch_done_events:
            assert "filter_reasons" in e
            
        print("[OK] test_deepseek_fallback_on_consecutive_failures passed")

if __name__ == "__main__":
    print("Running V1 Refinements unit tests...")
    test_robust_json_loads()
    test_conditional_dependencies_inference()
    test_detect_weak_structured_rows()
    test_verify_semantic_coverage()
    asyncio.run(test_text_generator_refinements())
    asyncio.run(test_structured_stream_refinements())
    asyncio.run(test_concurrent_lock_contention())
    asyncio.run(test_deepseek_fallback_on_consecutive_failures())
    print("All V1 Refinements tests passed successfully!")
