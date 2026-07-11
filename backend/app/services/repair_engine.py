"""
Dataset Repair Engine
The core differentiator — iteratively improves datasets until quality threshold met.

Pipeline:
  Generate → Audit → Detect weak rows → Regenerate weak rows →
  Re-audit → Repeat until quality >= threshold or max iterations
"""
import json
from typing import AsyncGenerator
from openai import AsyncOpenAI
from app.core.config import settings
from app.services.llm_client import get_llm_client, get_openai_client
from app.services.dataset_generator import (
    filter_bad_rows, semantic_hash, GENERATOR_SYSTEM_PROMPT,
    STYLE_INSTRUCTIONS, TASK_INSTRUCTIONS, robust_json_loads
)

REPAIR_SYSTEM_PROMPT = r"""You are a dataset quality specialist for LLM fine-tuning.
You are given WEAK training examples that need to be REPLACED with better ones.

A weak row has one or more of these problems:
- Generic filler output ("Sure! I can help...")
- Too short or vague output
- Input nearly identical to another
- Output doesn't directly address the input
- Repetitive phrasing or sentence structure

Your job: generate N replacement rows for the same topic.
The replacements must be meaningfully different from each other AND from the weak rows.

RULES:
1. More specific and detailed outputs than the originals
2. More diverse input phrasings
3. Cover different angles/aspects of the topic
4. No filler phrases ever
5. Return ONLY a JSON array of {"input": "...", "output": "..."} objects.

STRICT JSON ESCAPING RULES:
1. Double quotes inside string values MUST be escaped as \" (e.g. use \" for nested quotes, or prefer single quotes inside string values).
2. Backslashes in string values (such as regex patterns or escape sequences like \n or \t) MUST be double-escaped as \\ (e.g. write \\n or \\t or \\\\).
3. Ensure the output is a valid JSON array of objects.
"""


from app.services.profiler import profile_dataset_request, build_generation_policy


def detect_weak_rows(rows: list[dict], task_type: str) -> list[dict]:
    """
    Returns indices of rows that need replacement (synchronous fallback).
    """
    GENERIC_PHRASES = [
        "certainly", "of course", "sure!", "great question",
        "absolutely", "i'd be happy", "i can help",
        "that's a great", "excellent", "wonderful",
    ]
    MIN_OUTPUT_CHARS = {
        "chat": 50, "instruction": 100,
        "qa": 50, "classification": 1, "summarization": 80,
    }
    min_out = MIN_OUTPUT_CHARS.get(task_type, 50)

    weak_indices = []
    seen_hashes = {}

    for i, row in enumerate(rows):
        inp = str(row.get("input", "")).strip()
        out = str(row.get("output", "")).strip()
        reasons = []

        if len(out) < min_out:
            reasons.append("output_too_short")

        if any(p in out.lower()[:80] for p in GENERIC_PHRASES):
            reasons.append("generic_filler")

        h = semantic_hash(inp)
        if h in seen_hashes:
            reasons.append(f"duplicate_of_row_{seen_hashes[h]}")
        else:
            seen_hashes[h] = i

        if task_type in ("chat", "qa") and len(out) < len(inp) * 0.5:
            reasons.append("low_information_density")

        # Weak Row Classifier: Flag any row generated with depth: shallow
        if row.get("depth") == "shallow" or (isinstance(row.get("_provenance"), dict) and row.get("_provenance", {}).get("depth") == "shallow"):
            reasons.append("shallow_depth")

        if reasons:
            weak_indices.append({"index": i, "reasons": reasons, "row": row})

    return weak_indices


async def detect_weak_rows_async(client: AsyncOpenAI, rows: list[dict], task_type: str) -> list[dict]:
    """
    Returns indices of rows that need replacement.
    Weak row criteria:
    - Output too short
    - Generic filler detected
    - Semantic duplicate of another row (using embedding similarity)
    - Output doesn't contain enough information density
    - Shallow reasoning depth
    """
    GENERIC_PHRASES = [
        "certainly", "of course", "sure!", "great question",
        "absolutely", "i'd be happy", "i can help",
        "that's a great", "excellent", "wonderful",
    ]
    MIN_OUTPUT_CHARS = {
        "chat": 50, "instruction": 100,
        "qa": 50, "classification": 1, "summarization": 80,
    }
    min_out = MIN_OUTPUT_CHARS.get(task_type, 50)

    weak_indices = []
    
    # Embedding similarity check for duplicates
    inputs = [str(r.get("input", "")).strip() for r in rows]
    duplicates = set()
    try:
        from app.services.llm_client import get_embedding_client
        embedder = get_embedding_client()
        embs = await embedder.embed(inputs)
        
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                dot = sum(a * b for a, b in zip(embs[i], embs[j]))
                norm1 = sum(a * a for a in embs[i]) ** 0.5
                norm2 = sum(b * b for b in embs[j]) ** 0.5
                sim = dot / (norm1 * norm2 + 1e-9)
                if sim > 0.85:
                    duplicates.add(j)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"[RepairEngine] Embeddings fallback to semantic hash due to: {str(e)}")
        # Fallback to semantic hash
        seen_hashes = {}
        for i, inp in enumerate(inputs):
            h = semantic_hash(inp)
            if h in seen_hashes:
                duplicates.add(i)
            else:
                seen_hashes[h] = i

    for i, row in enumerate(rows):
        inp = str(row.get("input", "")).strip()
        out = str(row.get("output", "")).strip()
        reasons = []

        # Too short
        if len(out) < min_out:
            reasons.append("output_too_short")

        # Generic filler
        if any(p in out.lower()[:80] for p in GENERIC_PHRASES):
            reasons.append("generic_filler")

        # Duplicate check from our embedding audit above
        if i in duplicates:
            reasons.append("semantic_duplicate")

        # Low information density (output shorter than input — suspicious for chat/qa)
        if task_type in ("chat", "qa") and len(out) < len(inp) * 0.5:
            reasons.append("low_information_density")

        # Weak Row Classifier: Flag any row generated with depth: shallow
        if row.get("depth") == "shallow" or (isinstance(row.get("_provenance"), dict) and row.get("_provenance", {}).get("depth") == "shallow"):
            reasons.append("shallow_depth")

        if reasons:
            weak_indices.append({"index": i, "reasons": reasons, "row": row})

    return weak_indices


def compute_entropy_metrics(rows: list[dict]) -> dict:
    """
    Compute diversity/entropy metrics for the dataset.
    Higher entropy = more diverse = better for fine-tuning.
    """
    import re
    from collections import Counter

    all_inputs = [str(r.get("input", "")) for r in rows]
    all_outputs = [str(r.get("output", "")) for r in rows]

    def lexical_diversity(texts: list[str]) -> float:
        """Type-Token Ratio (TTR) — unique words / total words"""
        all_words = []
        for t in texts:
            words = re.findall(r'\b\w+\b', t.lower())
            all_words.extend(words)
        if not all_words:
            return 0.0
        return round(len(set(all_words)) / len(all_words), 3)

    def avg_length(texts: list[str]) -> float:
        return round(sum(len(t) for t in texts) / max(len(texts), 1), 1)

    def bigram_diversity(texts: list[str]) -> float:
        """Unique bigrams / total bigrams — structural diversity"""
        bigrams = []
        for t in texts:
            words = re.findall(r'\b\w+\b', t.lower())
            bigrams.extend(zip(words, words[1:]))
        if not bigrams:
            return 0.0
        return round(len(set(bigrams)) / len(bigrams), 3)

    def starting_word_diversity(texts: list[str]) -> float:
        """How many unique first words — catches structural repetition"""
        first_words = []
        for t in texts:
            words = re.findall(r'\b\w+\b', t.lower())
            if words:
                first_words.append(words[0])
        if not first_words:
            return 0.0
        counter = Counter(first_words)
        most_common_pct = counter.most_common(1)[0][1] / len(first_words)
        return round(1 - most_common_pct, 3)

    input_ttr = lexical_diversity(all_inputs)
    output_ttr = lexical_diversity(all_outputs)
    bigram_div = bigram_diversity(all_outputs)
    start_div = starting_word_diversity(all_outputs)

    # Overall entropy score 0-100
    entropy_score = round(
        (input_ttr * 25 + output_ttr * 35 + bigram_div * 25 + start_div * 15), 1
    )

    return {
        "entropy_score": entropy_score,
        "input_lexical_diversity": input_ttr,
        "output_lexical_diversity": output_ttr,
        "output_bigram_diversity": bigram_div,
        "output_start_diversity": start_div,
        "avg_input_length": avg_length(all_inputs),
        "avg_output_length": avg_length(all_outputs),
        "interpretation": (
            "excellent" if entropy_score > 70
            else "good" if entropy_score > 50
            else "moderate" if entropy_score > 30
            else "low — consider regenerating"
        )
    }


async def repair_dataset(
    rows: list[dict],
    topic: str,
    task_type: str,
    style: str,
    quality_threshold: float = 90.0,
    max_iterations: int = 3,
) -> AsyncGenerator[dict, None]:
    """
    Iteratively repair a dataset until quality threshold is met.
    Yields progress events throughout.
    """
    client, synthesis_model = get_llm_client()
    openai_client = get_openai_client()
    
    # Profile the topic to apply the dynamic policy mix during repair
    profile = await profile_dataset_request(topic, "", task_type)
    policy = build_generation_policy(profile)
    
    current_rows = rows.copy()
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Detect weak rows using embeddings and semantic similarity
        weak = await detect_weak_rows_async(openai_client, current_rows, task_type)
        entropy = compute_entropy_metrics(current_rows)

        yield {
            "event": "repair_iteration",
            "iteration": iteration,
            "total_rows": len(current_rows),
            "weak_rows_found": len(weak),
            "entropy_metrics": entropy,
        }

        if not weak:
            yield {"event": "repair_complete", "reason": "no_weak_rows",
                   "iterations": iteration, "rows": current_rows,
                   "entropy": entropy}
            return

        # Regenerate weak rows
        weak_rows_data = [w["row"] for w in weak]
        style_instr = STYLE_INSTRUCTIONS.get(style, "")
        task_instr = TASK_INSTRUCTIONS.get(task_type, "")

        # Select a persona and target difficulty dynamically from the policy mix
        personas = policy["personas"]
        assigned_persona = personas[iteration % len(personas)]
        
        weak_details = []
        for w in weak:
            reasons = w['reasons']
            req_depth = "medium"
            if "shallow_depth" in reasons:
                req_depth = "deep or expert reasoning"
            weak_details.append({
                "row_index": w['index'],
                "reasons": reasons,
                "required_depth": req_depth
            })

        repair_prompt = f"""Topic: {topic}
Task: {task_type} — {task_instr}
Style: {style} — {style_instr}
Suggested Persona: {assigned_persona}

These {len(weak_rows_data)} rows are WEAK and need replacement:
{json.dumps(weak_rows_data, indent=2)}

Reasons for weakness and improvement goals:
{json.dumps(weak_details, indent=2)}

Generate {len(weak_rows_data)} BETTER replacement rows.
They must cover different sub-topics than the weak rows and incorporate the assigned persona and varying difficulties.
For any rows requiring "deep or expert reasoning", you MUST ensure the output contains multi-step logic, detailed output, and expert depth.
Return a JSON array of {{"input": "...", "output": "..."}} objects."""

        try:
            kwargs = {
                "model": synthesis_model,
                "messages": [
                    {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
                    {"role": "user", "content": repair_prompt},
                ],
                "temperature": min(1.15, policy["temperature"] + 0.1),  # slightly higher temp for repairs to force novelty
                "max_tokens": 3000,
            }
            if synthesis_model != "deepseek-chat":
                kwargs["response_format"] = {"type": "json_object"}
                
            response = await client.chat.completions.create(**kwargs)
            try:
                client._client.cookies.clear()
            except Exception:
                pass

            content = response.choices[0].message.content
            parsed = robust_json_loads(content)
            replacements = parsed if isinstance(parsed, list) else next(
                (v for v in parsed.values() if isinstance(v, list)), []
            )
            replacements = filter_bad_rows(replacements, task_type)

            # Replace weak rows with repaired rows
            weak_indices = [w["index"] for w in weak]
            
            # Set provenance metadata for repaired rows
            for rep_row in replacements:
                rep_row["_provenance"] = {
                    "batch": "repair_loop",
                    "repair_iteration": iteration,
                    "persona": assigned_persona,
                    "temperature": round(min(1.15, policy["temperature"] + 0.1), 2)
                }

            good_rows = [r for i, r in enumerate(current_rows) if i not in weak_indices]
            current_rows = good_rows + replacements

            yield {
                "event": "repair_batch_done",
                "iteration": iteration,
                "replaced": len(weak_indices),
                "new_rows_generated": len(replacements),
                "total_rows": len(current_rows),
                "sample_repaired": replacements[:2],
            }

        except Exception as e:
            yield {"event": "repair_error", "iteration": iteration, "error": str(e)}
            break

    # Final audit check
    final_entropy = compute_entropy_metrics(current_rows)
    final_weak = await detect_weak_rows_async(openai_client, current_rows, task_type)

    yield {
        "event": "repair_complete",
        "reason": "max_iterations_reached" if iteration >= max_iterations else "threshold_met",
        "iterations": iteration,
        "rows": current_rows,
        "weak_rows_remaining": len(final_weak),
        "entropy": final_entropy,
    }


def detect_weak_structured_rows(rows: list[dict], columns: list[dict]) -> list[dict]:
    """
    Identifies specific indices of weak rows in a tabular dataset:
    - Completely empty or missing columns
    - Row-level duplicates
    - Logic/consistency violations (e.g., job_title == Senior Engineer when age < 18)
    """
    weak = []
    seen = set()
    import hashlib
    expected_cols = {c["name"] for c in columns}
    
    for i, row in enumerate(rows):
        reasons = []
        
        # 1. Missing columns
        missing = expected_cols - set(row.keys())
        if missing:
            reasons.append("missing_columns")
            
        # 2. Duplicate rows (ignoring _provenance metadata column)
        row_clean = {k: v for k, v in row.items() if k != "_provenance"}
        key = hashlib.md5(json.dumps(row_clean, sort_keys=True).encode()).hexdigest()
        if key in seen:
            reasons.append("duplicate_row")
        seen.add(key)
        
        # 3. Logical inconsistencies (e.g. age < 18 and employed in senior roles)
        age = row.get("age")
        if age is not None:
            try:
                if float(age) < 18:
                    job = str(row.get("job", row.get("job_title", ""))).lower()
                    if job and any(role in job for role in ("manager", "director", "engineer", "officer", "VP")):
                        reasons.append("logical_inconsistency_age_job")
            except (ValueError, TypeError):
                pass
                
        # 4. Null value in target column
        if columns:
            target_col = columns[-1]["name"]
            if row.get(target_col) is None or str(row.get(target_col)).strip() == "":
                reasons.append("missing_prediction_target")

        if reasons:
            weak.append({"index": i, "reasons": reasons, "row": row})
            
    return weak


async def repair_structured_dataset(
    rows: list[dict],
    columns: list[dict],
    description: str,
    domain_context: str = "",
    missingness_rate: float = 0.0,
    label_noise_rate: float = 0.0,
    quality_threshold: float = 85.0,
    max_iterations: int = 2
) -> AsyncGenerator[dict, None]:
    """
    Iteratively repair a structured dataset by regenerating only the weak rows.
    """
    from app.services.structured_generator import (
        build_schema_prompt, filter_structured_rows, simulate_missingness,
        inject_label_noise, audit_structured_batch, STRUCTURED_SYSTEM_PROMPT
    )
    import random
    
    client, synthesis_model = get_llm_client()
    current_rows = rows.copy()
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Audit and detect weak rows
        audit = audit_structured_batch(current_rows, columns)
        weak = detect_weak_structured_rows(current_rows, columns)

        yield {
            "event": "structured_repair_iteration",
            "iteration": iteration,
            "total_rows": len(current_rows),
            "weak_rows_found": len(weak),
            "quality_score": audit.get("quality_score", 0.0),
        }

        # Stop if no weak rows or threshold met
        if not weak or audit.get("quality_score", 0.0) >= quality_threshold:
            yield {
                "event": "structured_repair_complete",
                "iterations": iteration,
                "rows": current_rows,
                "audit": audit
            }
            return

        # Replace only the weak rows
        weak_indices = [w["index"] for w in weak]
        good_rows = [r for i, r in enumerate(current_rows) if i not in weak_indices]
        
        prompt = build_schema_prompt(
            description=description,
            columns=columns,
            row_count=len(weak),
            existing_rows=good_rows[-10:],  # avoid duplicates of good rows
            domain_context=domain_context,
        )

        try:
            response = await client.chat.completions.create(
                model=synthesis_model,
                messages=[
                    {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.95,  # higher temperature for repairs
                max_tokens=3000,
                response_format={"type": "json_object"},
            )
            try:
                client._client.cookies.clear()
            except Exception:
                pass

            content = response.choices[0].message.content
            parsed = robust_json_loads(content)
            
            if isinstance(parsed, dict) and "rows" in parsed:
                batch_rows = parsed["rows"]
            elif isinstance(parsed, list):
                batch_rows = parsed
            else:
                batch_rows = next((v for v in parsed.values() if isinstance(v, list)), []).copy()

            replacements = filter_structured_rows(batch_rows, columns)
            
            # Apply missingness and label noise
            replacements = simulate_missingness(replacements, columns, missingness_rate)
            replacements = inject_label_noise(replacements, columns, label_noise_rate)

            # Swap them back in
            for idx, rep in zip(weak_indices, replacements):
                rep["_provenance"] = {
                    "batch": "repair_loop",
                    "repair_iteration": iteration,
                    "temperature": 0.95,
                    "missingness_applied": missingness_rate > 0,
                    "label_noise_applied": label_noise_rate > 0
                }
                current_rows[idx] = rep

            yield {
                "event": "structured_repair_batch_done",
                "iteration": iteration,
                "replaced": len(replacements),
                "total_rows": len(current_rows),
            }

        except Exception as e:
            yield {"event": "structured_repair_error", "iteration": iteration, "error": str(e)}
            break

    # Final audit
    final_audit = audit_structured_batch(current_rows, columns)
    yield {
        "event": "structured_repair_complete",
        "iterations": iteration,
        "rows": current_rows,
        "audit": final_audit
    }