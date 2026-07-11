import sys
import os
import asyncio

# Add parent directory to path so app can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.dataset_generator import calculate_shannon_entropy, semantic_hash, filter_bad_rows, audit_generated_batch_async
from app.services.profiler import build_generation_policy
from app.services.structured_generator import simulate_missingness, inject_label_noise, audit_structured_batch

def test_calculate_shannon_entropy():
    # Equal distribution: entropy = - (4 * (0.25 * log2(0.25))) = 2.0
    labels = ["a", "b", "c", "d"]
    assert calculate_shannon_entropy(labels) == 2.0, "Entropy of equal distribution should be 2.0"

    # Single label: entropy = - (1 * log2(1.0)) = 0.0
    labels_single = ["a", "a", "a"]
    assert calculate_shannon_entropy(labels_single) == 0.0, "Entropy of single label should be 0.0"

    # Empty list: entropy = 0.0
    assert calculate_shannon_entropy([]) == 0.0, "Entropy of empty list should be 0.0"
    print("[OK] test_calculate_shannon_entropy passed")

def test_semantic_hash():
    # Hashing is case-insensitive and ignores punctuation
    text1 = "Hello, World! This is a test."
    text2 = "hello world this is a test"
    assert semantic_hash(text1) == semantic_hash(text2), "Semantic hashes should be identical"

    # Different initial words should produce different hashes
    text3 = "Different start word hello world this is a test"
    assert semantic_hash(text1) != semantic_hash(text3), "Semantic hashes should differ"
    print("[OK] test_semantic_hash passed")

def test_filter_bad_rows():
    # Rows with generic filler or too short outputs should be filtered out
    rows = [
        {"input": "Valid input query", "output": "This output is long enough and contains useful substantive content without filler."},
        {"input": "Query about filler", "output": "Certainly! I'd be happy to help you with that."}, # generic filler
        {"input": "Short output query", "output": "Short"}, # too short
        {"input": "Valid input query", "output": "This output is long enough and contains useful substantive content without filler."} # duplicate input
    ]
    
    clean = filter_bad_rows(rows, "chat")
    assert len(clean) == 1, f"Expected 1 clean row, got {len(clean)}"
    assert clean[0]["input"] == "Valid input query"
    print("[OK] test_filter_bad_rows passed")

def test_build_generation_policy():
    profile = {
        "factuality_risk": "high",
        "hallucination_sensitivity": "high",
        "reasoning_complexity": "high",
        "style_entropy_needed": "low",
        "edge_case_ratio": 0.20,
        "primary_personas": ["technical specialist", "expert analyst"],
        "dataset_name": "Test dataset name"
    }
    policy = build_generation_policy(profile)
    
    # High factuality risk + low style entropy should lower the temperature
    assert policy["temperature"] < 0.85, "Temperature should be lowered"
    assert policy["factuality_risk"] == "high"
    assert policy["dataset_name"] == "Test dataset name"
    assert "expert" in policy["reasoning_depths"]
    print("[OK] test_build_generation_policy passed")

def test_simulate_missingness():
    # Phase 2 Test: Verify missingness injection
    columns = [
        {"name": "feature_a", "type": "int"},
        {"name": "feature_b", "type": "string"},
        {"name": "target", "type": "category"} # target is the last column
    ]
    rows = [{"feature_a": 10, "feature_b": "hello", "target": "yes"} for _ in range(100)]
    
    corrupted = simulate_missingness(rows, columns, rate=0.20)
    
    # Check that the target column has NO missing values (unaffected target policy)
    target_vals = [r["target"] for r in corrupted]
    assert all(v == "yes" for v in target_vals), "Target column must never contain missing values"
    
    # Check that other columns contain some None or empty string values
    missing_count = sum(1 for r in corrupted if r["feature_a"] in (None, "") or r["feature_b"] in (None, ""))
    assert missing_count > 0, "Missing values should be successfully injected"
    print("[OK] test_simulate_missingness passed")

def test_inject_label_noise():
    # Phase 2 Test: Verify label noise injection
    columns = [
        {"name": "feature_a", "type": "int"},
        {"name": "target", "type": "category"}
    ]
    
    # Categorical flip test
    rows_cat = [{"feature_a": i, "target": "A" if i % 2 == 0 else "B"} for i in range(100)]
    corrupted_cat = inject_label_noise(rows_cat, columns, rate=0.15)
    
    # Count flipped values
    changed_cat = sum(1 for i, r in enumerate(corrupted_cat) if r["target"] != ("A" if i % 2 == 0 else "B"))
    assert changed_cat > 0, "Categorical label noise must flip some values"
    
    # Numeric target test
    columns_num = [
        {"name": "feature_a", "type": "int"},
        {"name": "target", "type": "float"}
    ]
    rows_num = [{"feature_a": i, "target": float(i * 10)} for i in range(100)]
    corrupted_num = inject_label_noise(rows_num, columns_num, rate=0.15)
    
    changed_num = sum(1 for i, r in enumerate(corrupted_num) if r["target"] != float(i * 10))
    assert changed_num > 0, "Numeric label noise must modify float values"
    print("[OK] test_inject_label_noise passed")

def test_target_leakage_and_realism():
    # Phase 2 Test: Verify target leakage audit and synthetic realism score calculation
    columns = [
        {"name": "leaking_feature", "type": "int"},
        {"name": "target", "type": "int"}
    ]
    
    # Deterministic columns: perfect correlation (r = 1.0)
    rows_leaked = [{"leaking_feature": i, "target": i} for i in range(20)]
    audit_leaked = audit_structured_batch(rows_leaked, columns)
    
    assert audit_leaked["target_leakage_found"], "Perfect correlation should trigger target leakage flag"
    assert audit_leaked["synthetic_realism_score"] < 80.0, "Leakage should penalize the synthetic realism score"
    print("[OK] test_target_leakage_and_realism passed")

async def test_readiness_grading_async():
    # Phase 2 Test: Verify A-F Fine-Tuning Readiness Grades
    
    # Case 1: Excellent quality dataset (high variance, no duplicates/contradictions)
    starters_in = ["Can you explain", "What is the difference for", "How does it work with", "Explain the concept of", "Describe the role of", "Compare traditional to"]
    starters_out = ["Detailing", "Explaining", "Analyzing", "Comparing", "Reviewing", "Describing"]
    concepts = [
        "quantum computing", "neural networks", "photosynthesis", "plate tectonics", "roman empire",
        "blockchain tech", "microservices", "genetic drift", "monetary policy", "renaissance art",
        "dark matter", "cellular respiration", "supply chain", "thermodynamics", "french revolution",
        "cryptography", "behavioral economics", "platelet activation", "stellar evolution", "gothic architecture",
        "game theory", "quantum entanglement", "immune response", "carbon cycle", "industrial revolution",
        "symbolic logic", "marine biology", "platelet aggregation", "capital structure", "baroque music"
    ]
    excellent_rows = []
    for i in range(30):
        s_in = starters_in[i % len(starters_in)]
        s_out = starters_out[i % len(starters_out)]
        concept = concepts[i]
        excellent_rows.append({
            "input": f"{s_in} {concept}?",
            "output": f"{s_out} detailed explanation of {concept} which explores its key aspects and significance."
        })
    audit_ex = await audit_generated_batch_async(None, excellent_rows, "chat")
    assert audit_ex["readiness_grade"] == "A", f"Expected grade A, got {audit_ex['readiness_grade']}"
    
    # Case 2: Highly repetitive dataset (duplicates and short responses)
    repetitive_rows = [
        {"input": "Duplicate query", "output": "Short response"}
        for _ in range(30)
    ]
    audit_rep = await audit_generated_batch_async(None, repetitive_rows, "chat")
    assert audit_rep["readiness_grade"] in ("D", "F"), f"Expected grade D or F, got {audit_rep['readiness_grade']}"
    print("[OK] test_readiness_grading_async passed")

if __name__ == "__main__":
    print("Running unit tests...")
    try:
        test_calculate_shannon_entropy()
        test_semantic_hash()
        test_filter_bad_rows()
        test_build_generation_policy()
        test_simulate_missingness()
        test_inject_label_noise()
        test_target_leakage_and_realism()
        asyncio.run(test_readiness_grading_async())
        print("All tests passed successfully!")
    except AssertionError as e:
        print(f"Test failure: {e}")
        sys.exit(1)
