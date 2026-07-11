import sys
import os
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.evaluation import EvaluationJob, EvaluationStatus
from app.models.dataset import Dataset
from app.models.job import TrainingJob
from app.services.evaluator import (
    clean_and_tokenize,
    calculate_rouge_n,
    calculate_rouge_l,
    calculate_semantic_similarity,
    generate_mock_responses,
    evaluate_model,
    calculate_fallback_judge_score,
    grade_with_llm_judge,
)

# 1. Test Tokenization and ROUGE-N/L Metric calculations
def test_metrics_calculation():
    ref = "the quick brown fox jumps over the lazy dog"
    cand = "the fast brown fox jumps over the lazy dog"
    
    ref_tokens = clean_and_tokenize(ref)
    cand_tokens = clean_and_tokenize(cand)
    
    assert ref_tokens == ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    assert cand_tokens == ["the", "fast", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    
    # ROUGE-1 F1 should be high (8 out of 9 overlap)
    p, r, f1_r1 = calculate_rouge_n(ref_tokens, cand_tokens, n=1)
    assert abs(f1_r1 - 8/9) < 1e-5
    
    # ROUGE-2 F1 (2-gram overlaps)
    # Ref 2-grams: 8 total. Cand 2-grams: 8 total. Overlap: 6.
    p, r, f1_r2 = calculate_rouge_n(ref_tokens, cand_tokens, n=2)
    assert abs(f1_r2 - 0.75) < 1e-5
    
    # ROUGE-L LCS F1
    p, r, f1_rl = calculate_rouge_l(ref_tokens, cand_tokens)
    assert abs(f1_rl - 8/9) < 1e-5
    
    print("[OK] test_metrics_calculation passed")

# 2. Test Semantic Jaccard Fallback
async def test_semantic_similarity_fallback():
    # Empty setting or placeholder key triggers fallback Jaccard
    with patch("app.services.evaluator.settings") as mock_settings:
        mock_settings.openai_api_key = "placeholder"
        sim = await calculate_semantic_similarity("hello world", "hello there world")
        # Jaccard of {"hello", "world"} and {"hello", "there", "world"} -> 2/3
        assert abs(sim - 2/3) < 1e-2
        
    print("[OK] test_semantic_similarity_fallback passed")

# 3. Test Mock Generation
async def test_mock_responses():
    with patch("app.services.evaluator.settings") as mock_settings:
        mock_settings.openai_api_key = None
        base, ft, base_lat, ft_lat = await generate_mock_responses("What is 2+2?", "Four")
        assert "base" in base.lower() or "certainly" in base.lower()
        assert ft == "Four"
        assert 700 <= base_lat <= 1300
        assert 300 <= ft_lat <= 650
        
    print("[OK] test_mock_responses passed")

# 4. Test evaluate_model full loop with mock database session
async def test_evaluate_model_pipeline():
    eval_job_id = "test-eval-job"
    dataset_id = "test-dataset-id"
    
    # Mock evaluation job
    mock_job = EvaluationJob(
        id=eval_job_id,
        user_id="user-123",
        training_job_id="job-123",
        dataset_id=dataset_id,
        status=EvaluationStatus.queued,
        metrics=None,
        comparison_data=None,
    )
    
    # Mock dataset containing rows
    mock_dataset = Dataset(
        id=dataset_id,
        user_id="user-123",
        name="Test DS",
        status="ready",
        audit_results={
            "rows": [
                {"input": "Describe the sun", "output": "The sun is a massive star at the center of the solar system."},
                {"input": "What is water?", "output": "Water is H2O, a chemical compound of hydrogen and oxygen."}
            ]
        }
    )
    
    # Mock db session execute returns
    mock_session = AsyncMock()
    mock_session.__aenter__.return_value = mock_session
    
    class MockResult:
        def __init__(self, scalar_val):
            self._val = scalar_val
        def scalar_one_or_none(self):
            return self._val
            
    # Mock training job
    mock_training_job = TrainingJob(
        id="job-123",
        user_id="user-123",
        base_model="microsoft/phi-3-mini-4k-instruct",
        status="completed"
    )

    # First query is for EvaluationJob, second query is for Dataset, third query is for TrainingJob
    mock_session.execute.side_effect = [
        MockResult(mock_job),
        MockResult(mock_dataset),
        MockResult(mock_training_job)
    ]
    
    with patch("app.services.evaluator.AsyncSessionLocal", return_value=mock_session), \
         patch("asyncio.sleep", AsyncMock()): # fast execution
         
        res = await evaluate_model(eval_job_id)
        
        assert res["status"] == "completed"
        assert mock_job.status == EvaluationStatus.completed
        assert mock_job.metrics is not None
        assert "rouge_1" in mock_job.metrics
        assert "llm_judge_score" in mock_job.metrics
        assert "llm_judge_base_score" in mock_job.metrics
        assert 1.0 <= mock_job.metrics["llm_judge_score"] <= 5.0
        assert 1.0 <= mock_job.metrics["llm_judge_base_score"] <= 5.0
        assert "semantic_similarity" in mock_job.metrics
        assert "avg_latency_ms" in mock_job.metrics
        assert len(mock_job.comparison_data) == 2
        assert mock_job.comparison_data[0]["input"] == "Describe the sun"
        assert "llm_judge_score" in mock_job.comparison_data[0]["metrics"]
        assert "explanation" in mock_job.comparison_data[0]
        
    print("[OK] test_evaluate_model_pipeline passed")

# 5. Test LLM Judge Grading and Heuristic Fallbacks
async def test_llm_judge_and_fallback():
    # Test fallback calculator directly
    base, ft, explanation = calculate_fallback_judge_score(
        target="Water is H2O.",
        base_response="Water is a liquid.",
        ft_response="Water is H2O."
    )
    # Verify scores are floats between 1.0 and 5.0
    assert isinstance(base, float)
    assert 1.0 <= base <= 5.0
    assert isinstance(ft, float)
    assert 1.0 <= ft <= 5.0
    assert ft >= base  # FT has exact token match, should score higher or equal
    assert "fallback" in explanation.lower()

    # Test grade_with_llm_judge async fault tolerance
    mock_bad_client = MagicMock()
    mock_bad_client.chat.completions.create.side_effect = Exception("API rate limit exceeded")

    # Call grade_with_llm_judge and verify it catches the error and falls back gracefully
    base_score, ft_score, explanation = await grade_with_llm_judge(
        client=mock_bad_client,
        prompt="What is water?",
        target="Water is H2O.",
        base_response="Water is a liquid.",
        ft_response="Water is H2O."
    )

    assert isinstance(base_score, float)
    assert 1.0 <= base_score <= 5.0
    assert isinstance(ft_score, float)
    assert 1.0 <= ft_score <= 5.0
    assert "fallback" in explanation.lower()

    print("[OK] test_llm_judge_and_fallback passed")

if __name__ == "__main__":
    print("=== Running Model Evaluation Engine Unit tests ===")
    test_metrics_calculation()
    asyncio.run(test_semantic_similarity_fallback())
    asyncio.run(test_mock_responses())
    asyncio.run(test_evaluate_model_pipeline())
    asyncio.run(test_llm_judge_and_fallback())
    print("=== All Model Evaluation Engine tests passed! ===")
