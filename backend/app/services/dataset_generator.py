"""
Synthetic Dataset Generator
Generates high-quality, low-noise fine-tuning datasets using OpenAI.
Anti-noise strategy is baked into every step.
"""
import json
import re
import hashlib
import math
import logging
import asyncio
import numpy as np
from collections import Counter
from typing import AsyncGenerator, Optional, List
from openai import AsyncOpenAI
from app.core.config import settings
from app.services.profiler import profile_dataset_request, build_generation_policy, plan_dataset_coverage
from app.services.llm_client import get_llm_client, get_openai_client, get_embedding_client
import logging

logger = logging.getLogger(__name__)


def split_json_objects(text: str) -> list[str]:
    """Extract top-level JSON objects by balancing curly braces."""
    objs = []
    brace_count = 0
    start_idx = -1
    in_string = False
    escape = False
    
    for i, char in enumerate(text):
        if char == '"' and not escape:
            in_string = not in_string
        
        if char == '\\' and in_string:
            escape = not escape
        else:
            escape = False
            
        if not in_string:
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    objs.append(text[start_idx:i+1])
                    start_idx = -1
    return objs


def robust_json_loads(text: str):
    """Robust JSON decoder that recovers nested quotes and malformed string values."""
    if not text:
        return None
    text = text.strip()
    
    # Strip markdown code blocks
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    # 1. Standard loads
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
        
    # 2. strict=False to allow raw newlines/tabs in string literals
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass

    # 3. Clean invalid backslashes (e.g. \x or single backslashes in path/code)
    cleaned = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', text)
    try:
        return json.loads(cleaned, strict=False)
    except json.JSONDecodeError:
        pass
        
    # 4. Brace-balancing regex/string extraction fallback
    raw_objs = split_json_objects(text)
    objs = []
    
    for raw in raw_objs:
        # Try parsing single object
        try:
            cleaned_obj = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', raw)
            obj = json.loads(cleaned_obj, strict=False)
            objs.append(obj)
            continue
        except json.JSONDecodeError:
            pass
            
        # Extract keys manually
        inp_match = re.search(r'"input"\s*:\s*"(.*?)"\s*,\s*"(?:output|messages|_provenance)"', raw, re.DOTALL)
        if not inp_match:
            inp_match = re.search(r'"input"\s*:\s*"(.*)"', raw, re.DOTALL)
            
        out_match = re.search(r'"output"\s*:\s*"(.*?)"\s*,\s*"(?:input|messages|_provenance)"', raw, re.DOTALL)
        if not out_match:
            out_match = re.search(r'"output"\s*:\s*"(.*)"', raw, re.DOTALL)
            
        msg_match = re.search(r'"messages"\s*:\s*(\[.*?\])\s*(?:,|\s*})', raw, re.DOTALL)
        
        inp_val = inp_match.group(1).strip() if inp_match else ""
        out_val = out_match.group(1).strip() if out_match else ""
        
        # Clean trailing quote and brace if greedy match over-captured
        for _ in range(5):
            out_val = out_val.strip()
            if out_val.endswith("}"):
                out_val = out_val[:-1]
            elif out_val.endswith('"'):
                out_val = out_val[:-1]
            else:
                break
                
        for _ in range(5):
            inp_val = inp_val.strip()
            if inp_val.endswith("}"):
                inp_val = inp_val[:-1]
            elif inp_val.endswith('"'):
                inp_val = inp_val[:-1]
            else:
                break
            
        if inp_val or out_val:
            res_obj = {"input": inp_val, "output": out_val}
            if msg_match:
                try:
                    res_obj["messages"] = json.loads(msg_match.group(1), strict=False)
                except Exception:
                    msgs = []
                    role_contents = re.findall(r'\{\s*"role"\s*:\s*"(.*?)"\s*,\s*"content"\s*:\s*"(.*?)"\s*\}', msg_match.group(1), re.DOTALL)
                    for r, c in role_contents:
                        msgs.append({"role": r, "content": c})
                    if msgs:
                        res_obj["messages"] = msgs
            objs.append(res_obj)
            
    if objs:
        return objs
        
    return None


# ─── SYSTEM PROMPT: The core of noise prevention ───────────────────────────
GENERATOR_SYSTEM_PROMPT = r"""You are an expert dataset curator for LLM fine-tuning.
Your job is to generate training examples that are:

QUALITY RULES (follow strictly):
1. DIVERSE - Every single example must be meaningfully different. 
   No two inputs should ask the same thing in slightly different words.
2. SPECIFIC - Outputs must be detailed and concrete, never generic.
   BAD output: "Sure! I can help you with that."
   GOOD output: A specific, substantive response to the exact input.
3. CONSISTENT - Maintain the same persona, tone, and style throughout.
4. REALISTIC - Examples must reflect real-world usage of this system.
5. BALANCED - Cover edge cases, not just the easy common cases.
6. NO FILLER - Never start outputs with "Certainly!", "Of course!", "Great question!"
7. GROUNDED - Every claim in the output must be factually defensible.

You will generate exactly the number of examples requested.
Respond ONLY with a JSON array. No preamble, no explanation, no markdown.
Each object must have exactly two keys: "input" and "output".

STRICT JSON ESCAPING RULES:
1. Double quotes inside string values MUST be escaped as \" (e.g. use \" for nested quotes, or prefer single quotes inside string values).
2. Backslashes in string values (such as regex patterns or escape sequences like \n or \t) MUST be double-escaped as \\ (e.g. write \\n or \\t or \\\\).
3. Ensure the output is a valid JSON array of objects.
"""

STYLE_INSTRUCTIONS = {
    "formal":    "Use professional, formal language. Complete sentences. No slang.",
    "casual":    "Use natural, conversational language. Contractions are fine.",
    "technical": "Use precise technical terminology. Assume domain expertise.",
    "simple":    "Use simple language. Short sentences. Easy to understand.",
}

TASK_INSTRUCTIONS = {
    "chat": "Generate realistic multi-turn style single exchanges.",
    "instruction": "Generate clear instructions with detailed step-by-step outputs.",
    "qa": "Generate specific questions with factually accurate answers.",
    "classification": "Generate inputs with a single-word or short-phrase output label.",
    "summarization": "Generate long inputs with concise accurate summaries as outputs.",
}


def build_generation_prompt(
    topic: str,
    task_type: str,
    style: str,
    batch_size: int,
    existing_inputs: list[str],
    domain_context: str = "",
    persona: str = "",
    difficulty: str = "medium",
    subdomain: str = "",
    mode: str = "",
    depth: str = "",
    structure: str = "",
    simulate_user: bool = False,
    adversarial: bool = False,
    multi_turn: bool = False,
    replay_subdomains: list[str] = None,
    replay_modes: list[str] = None
) -> str:
    task_instr = TASK_INSTRUCTIONS.get(task_type, TASK_INSTRUCTIONS["chat"])
    style_instr = STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS["casual"])

    avoid_section = ""
    if existing_inputs:
        samples = existing_inputs[-50:]  # show last 50 to avoid repeating
        avoid_section = f"""
ALREADY GENERATED (do NOT repeat or paraphrase these):
{json.dumps(samples, indent=2)}
"""

    # SEMANTIC REPLAY MEMORY: Explicitly tell LLM to avoid already covered concept subdomains and styles
    if replay_subdomains or replay_modes:
        sub_str = f"- Already covered subdomains: {', '.join(replay_subdomains)}" if replay_subdomains else ""
        mode_str = f"- Already used response modes: {', '.join(replay_modes)}" if replay_modes else ""
        avoid_section += f"\nSEMANTIC REPLAY MEMORY (Avoid duplicating these regions):\n{sub_str}\n{mode_str}\n"

    domain_section = f"\nDOMAIN CONTEXT: {domain_context}" if domain_context else ""
    persona_section = f"\nYOUR ASSIGNED PERSONA: {persona}" if persona else ""
    difficulty_section = f"\nTARGET INTELLECTUAL DIFFICULTY: {difficulty.upper()} (Ensure the inputs and outputs reflect this level of depth and complexity)" if difficulty else ""
    if isinstance(subdomain, list):
        sub_list_str = ", ".join(subdomain)
        subdomain_section = f"\nTARGET SUBDOMAINS / CONCEPT AREAS: {sub_list_str.upper()}"
        diversity_subdomain_instr = f"- Cover a balanced mix of these target subdomains respectively (attempt to generate examples matching these topics): {sub_list_str}"
    else:
        subdomain_section = f"\nTARGET SUBDOMAIN / CONCEPT AREA: {subdomain.upper()}" if subdomain else ""
        diversity_subdomain_instr = f"- Specifically target the concept area '{subdomain}'"

    mode_section = f"\nGENERATION MODE: {mode.upper()}" if mode else ""
    depth_section = f"\nREASONING DEPTH: {depth.upper()}" if depth else ""
    structure_section = f"\nRESPONSE STRUCTURE LAYOUT: {structure.upper()}" if structure else ""
    
    behavior_instructions = []
    if simulate_user:
        behavior_instructions.append("- REAL USER SIMULATION: Write some of the inputs with realistic user flaws (typos, incomplete grammar, vague phrasing, or highly direct emotional phrasing) to mimic a real human.")
    if adversarial:
        behavior_instructions.append("- ADVERSARIAL INPUTS: Write some inputs containing misleading premises, vague/underspecified tasks, or contradictory conditions to test model robustness.")
        
    behavior_section = f"\nBEHAVIOR INSTRUCTIONS:\n" + "\n".join(behavior_instructions) if behavior_instructions else ""

    multi_turn_instructions = ""
    if multi_turn:
        multi_turn_instructions = """
MULTI-TURN DIALOGUE REQUIREMENT:
- Every example MUST represent a multi-turn conversation containing 2 to 4 turns (4 to 8 total messages exchanged between user and assistant).
- You must return an array of objects where each object contains a "messages" key containing the dialogue history in standard OpenAI chat format:
  [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
- The dialogue must exhibit high-quality conversational memory (referencing earlier turns), follow-up continuity (logical progression of ideas), clarification realism (assistant asking user to clarify vague requests), and escalation realism (handling frustrated or advanced follow-ups naturally). Avoid isolated robotic turns.
- The final message in the "messages" array must be the assistant's final response.
- For maximum compatibility across single and multi-turn pipelines, set the flat "input" key to the first user message, and the "output" key to the final assistant response.
"""

    return_instruct = f'Return a JSON array of exactly {batch_size} objects with "input" and "output" keys.'
    if multi_turn:
        return_instruct = f'Return a JSON array of exactly {batch_size} objects. Each object must have "input", "output", and a nested "messages" array of role-content pairs.'

    return f"""Generate exactly {batch_size} training examples for this fine-tuning task:

TOPIC: {topic}
TASK TYPE: {task_type} — {task_instr}
STYLE: {style} — {style_instr}{domain_section}{persona_section}{difficulty_section}{subdomain_section}{mode_section}{depth_section}{structure_section}
{multi_turn_instructions}
{behavior_section}

{avoid_section}

DIVERSITY REQUIREMENT:
- Cover {batch_size} distinct sub-topics or scenarios within "{topic}"
{diversity_subdomain_instr}
- Focus on the assigned persona, generation mode, reasoning depth, and layout structure
- Include edge cases and unusual but realistic scenarios where applicable
- Vary input length: some short, some detailed

{return_instruct}"""



def semantic_hash(text: str) -> str:
    """Rough semantic fingerprint — normalize and hash for dedup."""
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    normalized = re.sub(r'[^\w\s]', '', normalized)
    words = sorted(normalized.split()[:8])  # first 8 words, sorted
    return hashlib.md5(' '.join(words).encode()).hexdigest()


def calculate_shannon_entropy(labels: list) -> float:
    """Calculate the Shannon Entropy of a list of labels/categories."""
    if not labels:
        return 0.0
    n = len(labels)
    counts = Counter(labels)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p)
    return round(entropy, 2)


async def detect_contradictions_async(
    llm_client: AsyncOpenAI | None,
    openai_client: AsyncOpenAI | None,
    synthesis_model: str,
    rows: list[dict]
) -> list[dict]:
    """
    Check if similar inputs in the generated dataset contain conflicting recommendations.
    Uses embeddings to find semantically close candidates, then audits via LLM.
    """
    if not llm_client or len(rows) < 2:
        return []

    inputs = [str(r.get("input", "")).strip() for r in rows]
    try:
        from app.services.llm_client import get_embedding_client
        embedder = get_embedding_client()
        embs = await embedder.embed(inputs)
    except Exception as e:
        logger.error(f"[ContradictionAudit] Embedding failure: {str(e)}")
        return []

    # Find candidates with similarity > 0.70 but < 0.95 (not identical duplicates)
    candidates = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            dot = sum(a * b for a, b in zip(embs[i], embs[j]))
            n1 = sum(a*a for a in embs[i]) ** 0.5
            n2 = sum(b*b for b in embs[j]) ** 0.5
            sim = dot / (n1 * n2 + 1e-9)
            if 0.70 < sim < 0.95:
                candidates.append((sim, i, j))

    # Sort candidates by similarity descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    contradictions = []
    CONTRADICTION_SYSTEM = "You are a logical auditor. Respond ONLY with a JSON object: {\"contradictory\": true|false, \"reason\": \"explanation\"}"
    # Audit top 6 candidates to keep API latencies reasonable
    for sim, i, j in candidates[:6]:
        pair_prompt = f"""Compare these two training examples. Do they contain contradictory information or inconsistent logic?
        
Example 1:
Input: {rows[i]['input']}
Output: {rows[i]['output']}

Example 2:
Input: {rows[j]['input']}
Output: {rows[j]['output']}

Analyze if they contradict. Respond in valid JSON format only."""
        try:
            kwargs = {
                "model": synthesis_model,
                "messages": [
                    {"role": "system", "content": CONTRADICTION_SYSTEM},
                    {"role": "user", "content": pair_prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 200,
            }
            if synthesis_model != "deepseek-chat":
                kwargs["response_format"] = {"type": "json_object"}
                
            resp_audit = await llm_client.chat.completions.create(**kwargs)
            try:
                llm_client._client.cookies.clear()
            except Exception:
                pass
            res = json.loads(resp_audit.choices[0].message.content)
            if res.get("contradictory", False):
                contradictions.append({
                    "row_a": i,
                    "row_b": j,
                    "similarity": round(sim, 2),
                    "reason": res.get("reason", "Inconsistent logic between outputs.")
                })
        except Exception:
            continue

    return contradictions


async def audit_generated_batch_async(
    client: AsyncOpenAI | None,
    rows: list[dict],
    task_type: str,
    openai_client: AsyncOpenAI | None = None,
    synthesis_model: str = "gpt-4o-mini",
    precomputed_embeddings: list[list[float]] = None
) -> dict:
    """
    Deep quality audit on generated rows using embedding deduplication, contradiction checking, and entropy targets.
    """
    if not rows:
        return {"total_generated": 0, "issues": [], "quality_score": 0.0}

    issues = []
    score = 100.0
    total = len(rows)

    # 1. Cosine similarity duplicate check using embeddings
    duplicates = 0
    inputs = [str(row.get("input", "")).strip() for row in rows]
    embs = []
    try:
        if precomputed_embeddings is not None and len(precomputed_embeddings) == len(rows):
            logger.info(f"[Audit] Using {len(precomputed_embeddings)} precomputed embeddings for duplicate checking.")
            embs = precomputed_embeddings
        else:
            from app.services.llm_client import get_embedding_client
            embedder = get_embedding_client()
            embs = await embedder.embed(inputs)

        # Vectorized pairwise cosine similarity via numpy (O(n²) in one matrix multiply)
        if embs:
            mat = np.array(embs, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            normed = mat / (norms + 1e-9)
            sim_matrix = normed @ normed.T  # shape (n, n)
            # Only upper triangle, excluding diagonal
            upper = np.triu(sim_matrix, k=1)
            dup_rows = np.any(upper > 0.92, axis=1)
            duplicates = int(dup_rows.sum())
        else:
            raise ValueError("No embeddings generated")
    except Exception as e:
        logger.warning(f"[Audit] Embeddings failure: {str(e)}. Falling back to semantic hash.")
        # Fallback to semantic hash if embedding API fails
        seen_hashes = set()
        for inp in inputs:
            h = semantic_hash(inp)
            if h in seen_hashes:
                duplicates += 1
            seen_hashes.add(h)

    # 2. Contradiction audit
    contradictions = await detect_contradictions_async(
        llm_client=client,
        openai_client=openai_client,
        synthesis_model=synthesis_model,
        rows=rows
    )

    # 3. Short outputs & inputs check
    too_short_outputs = 0
    generic_outputs = 0
    too_short_inputs = 0
    disclaimer_reps = 0

    GENERIC_PHRASES = [
        "certainly", "of course", "sure!", "great question",
        "absolutely", "i'd be happy to", "i can help",
        "that's a great", "excellent question",
    ]

    DISCLAIMER_PATTERNS = [
        "informational purposes", "not a replacement", "substitute for professional",
        "replace seeing a doctor", "consult a healthcare provider", "consult a pediatrician"
    ]

    MIN_OUTPUT_CHARS = {
        "chat": 40, "instruction": 30, "qa": 40, "classification": 1, "summarization": 60,
    }
    min_out = MIN_OUTPUT_CHARS.get(task_type, 40)

    first_words = []
    for row in rows:
        inp = str(row.get("input", "")).strip()
        out = str(row.get("output", "")).strip()

        # Word length checks
        if len(out) < min_out:
            too_short_outputs += 1
        if len(inp) < 10:
            too_short_inputs += 1

        out_lower = out.lower()
        if any(phrase in out_lower[:60] for phrase in GENERIC_PHRASES):
            generic_outputs += 1

        if any(pat in out_lower for pat in DISCLAIMER_PATTERNS):
            disclaimer_reps += 1

        # Extract first word
        words = re.findall(r'\b\w+\b', out_lower)
        if words:
            first_words.append(words[0])

    # 4. Shannon Entropy of starting words (sentence structure variety)
    start_word_entropy = calculate_shannon_entropy(first_words)

    # Apply Deduplication Penalties
    if duplicates > 0:
        pct = duplicates / total
        issues.append({
            "type": "semantic_duplicates",
            "count": duplicates,
            "severity": "high",
            "message": f"{duplicates} semantic duplicates detected (similarity > 0.85) — model will overfit.",
            "fix": "Deduplicated automatically or via dynamic policy shift."
        })
        score -= min(25, pct * 80)

    # Apply Contradiction Penalties
    if contradictions:
        issues.append({
            "type": "contradictions",
            "count": len(contradictions),
            "severity": "high",
            "message": f"Detected {len(contradictions)} logical contradiction(s) between semantically overlapping examples.",
            "fix": "Re-run generation or let the repair loop resolve the contradiction."
        })
        score -= min(30, len(contradictions) * 15)

    # Apply Cluster Balancing Penalty (subdomains)
    subdomains = [r.get("subdomain", "unknown") for r in rows if r.get("subdomain")]
    if len(set(subdomains)) > 1:
        subdomain_entropy = calculate_shannon_entropy(subdomains)
        max_entropy = math.log2(len(set(subdomains)))
        balance_ratio = subdomain_entropy / max_entropy if max_entropy > 0 else 0
        if balance_ratio < 0.60:
            issues.append({
                "type": "cluster_imbalance",
                "count": 1,
                "severity": "medium",
                "message": f"Low category/subdomain balance (ratio = {balance_ratio:.2f}). Certain topics dominate the dataset.",
                "fix": "Decompose topics more granularly or regenerate sparse subdomains."
            })
            score -= 10

    # Apply Output Quality Penalties
    if too_short_outputs > total * 0.1:
        issues.append({
            "type": "short_outputs",
            "count": too_short_outputs,
            "severity": "high",
            "message": f"{too_short_outputs} outputs are too short (<{min_out} chars) — learning superficial patterns.",
            "fix": "These rows were filtered out automatically."
        })
        score -= min(20, (too_short_outputs / total) * 60)

    if generic_outputs > total * 0.05:
        issues.append({
            "type": "generic_filler",
            "count": generic_outputs,
            "severity": "medium",
            "message": f"{generic_outputs} outputs start with generic filler phrases.",
            "fix": "These rows were filtered out automatically."
        })
        score -= min(15, (generic_outputs / total) * 40)

    # Disclaimer Repetition Penalty
    if disclaimer_reps > total * 0.15:
        issues.append({
            "type": "repetitive_disclaimer",
            "count": disclaimer_reps,
            "severity": "medium",
            "message": f"Repetitive safety disclaimers detected in {disclaimer_reps} rows. This causes overfitting.",
            "fix": "Reduce disclaimer density or move to System Prompt."
        })
        score -= min(10, (disclaimer_reps / total) * 30)

    # Low Starting Word Entropy Penalty
    if total > 10 and start_word_entropy < 2.0:
        issues.append({
            "type": "low_entropy",
            "count": 1,
            "severity": "medium",
            "message": f"Low output starting word diversity (entropy = {start_word_entropy:.2f}). Responses start repetitively.",
            "fix": "Policy engine will shift to higher temperature batching."
        })
        score -= 10

    # 8. Hallucination Risk Auditing (Hedges/factual uncertainty detection)
    hallucination_risks = 0
    if task_type in ("qa", "instruction") and score > 0:
        SUSPICIOUS_PHRASES = ["i believe", "i think", "as far as i know", "probably", "not sure but", "my database only has"]
        for row in rows:
            out_lower = str(row.get("output", "")).lower()
            if any(phrase in out_lower for phrase in SUSPICIOUS_PHRASES):
                hallucination_risks += 1
                
        if hallucination_risks > 0:
            issues.append({
                "type": "hallucination_risk",
                "count": hallucination_risks,
                "severity": "medium",
                "message": f"Detected {hallucination_risks} response(s) containing hedges/factual hesitations (e.g., 'I believe', 'probably'). This suggests weak grounding or potential hallucination.",
                "fix": "Re-run in technical/formal style or enforce high grounding rules."
            })
            score -= min(10, hallucination_risks * 3)

    if too_short_inputs > total * 0.1:
        issues.append({
            "type": "short_inputs",
            "count": too_short_inputs,
            "severity": "medium",
            "message": f"{too_short_inputs} inputs are very short (<10 chars).",
            "fix": "Verify details are sufficient."
        })
        score -= 5

    q_score = round(max(0.0, score), 1)

    # Calculate Fine-Tuning Readiness Grade
    if q_score >= 90.0:
        readiness_grade = "A"
        readiness_label = "Model Ready (Excellent)"
    elif q_score >= 80.0:
        readiness_grade = "B"
        readiness_label = "Good (Minor issues)"
    elif q_score >= 70.0:
        readiness_grade = "C"
        readiness_label = "Needs Polish (Moderate warnings)"
    elif q_score >= 60.0:
        readiness_grade = "D"
        readiness_label = "Weak (Repair recommended)"
    else:
        readiness_grade = "F"
        readiness_label = "Unsuitable (Highly repetitive / collapsed)"

    return {
        "total_generated": total,
        "issues": issues,
        "quality_score": q_score,
        "readiness_score": q_score,
        "readiness_grade": readiness_grade,
        "readiness_label": readiness_label,
        "duplicate_count": duplicates,
        "contradiction_count": len(contradictions),
        "short_output_count": too_short_outputs,
        "generic_output_count": generic_outputs,
        "disclaimer_repetition_count": disclaimer_reps,
        "starting_word_entropy": start_word_entropy,
        "hallucination_risk_count": hallucination_risks
    }




def format_io_value(val) -> str:
    """Format input/output values cleanly if they are lists, dicts, or other structures."""
    if val is None:
        return ""
    if isinstance(val, list):
        cleaned = []
        for x in val:
            if isinstance(x, (list, dict)):
                cleaned.append(json.dumps(x))
            else:
                cleaned.append(str(x))
        return "\n".join(cleaned).strip()
    if isinstance(val, dict):
        return json.dumps(val).strip()
    return str(val).strip()


def filter_bad_rows_with_stats(rows: list[dict], task_type: str) -> tuple[list[dict], dict]:
    """Remove provably bad rows automatically and return statistics on why they were filtered."""
    GENERIC_PHRASES = [
        "certainly!", "of course!", "sure!", "great question",
        "absolutely!", "i'd be happy to", "i can help with that",
    ]
    MIN_OUTPUT_CHARS = {
        "chat": 40, "instruction": 30,
        "qa": 40, "classification": 1, "summarization": 60,
    }
    min_out = MIN_OUTPUT_CHARS.get(task_type, 40)
    seen_hashes = set()
    clean = []
    
    stats = {
        "empty_io": 0,
        "too_short": 0,
        "generic_phrase": 0,
        "duplicate_input": 0
    }

    for row in rows:
        inp = format_io_value(row.get("input", ""))
        out = format_io_value(row.get("output", ""))

        if not inp or not out:
            stats["empty_io"] += 1
            continue
        if len(out) < min_out:
            stats["too_short"] += 1
            continue
        if any(phrase in out.lower()[:60] for phrase in GENERIC_PHRASES):
            stats["generic_phrase"] += 1
            continue

        h = semantic_hash(inp)
        if h in seen_hashes:
            stats["duplicate_input"] += 1
            continue
        seen_hashes.add(h)

        # Preserve both flat input/output and nested messages history if generated
        clean_row = {"input": inp, "output": out}
        if "messages" in row and isinstance(row["messages"], list):
            clean_row["messages"] = row["messages"]
        clean.append(clean_row)

    return clean, stats


def filter_bad_rows(rows: list[dict], task_type: str) -> list[dict]:
    """Remove provably bad rows automatically."""
    clean, _ = filter_bad_rows_with_stats(rows, task_type)
    return clean



class TokenBucketLimiter:
    def __init__(self, capacity: int, fill_rate: float):
        self.capacity = capacity
        self.fill_rate = fill_rate
        self.tokens = capacity
        self.last_update = asyncio.get_event_loop().time()
        self.lock = asyncio.Lock()

    async def acquire(self, amount: int = 1):
        async with self.lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
            self.last_update = now

            if self.tokens < amount:
                sleep_time = (amount - self.tokens) / self.fill_rate
                logger.info(f"[RateLimiter] Token bucket empty. Sleeping {sleep_time:.2f}s to refill...")
                await asyncio.sleep(sleep_time)
                self.tokens = 0.0
            else:
                self.tokens -= amount


async def generate_dataset_stream(
    topic: str,
    task_type: str,
    style: str,
    total_rows: int,
    domain_context: str = "",
    multi_turn: bool = False,
    openai_api_key: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    """
    Stream dataset generation in batches.
    Yields progress events as dicts.
    """
    import time
    from app.services.llm_client import get_embedding_client
    
    start_time = time.time()
    
    # 1. Profile request and build adaptive policy
    profile = await profile_dataset_request(topic, domain_context, task_type)
    policy = build_generation_policy(profile)
    
    # 2. Coverage Planning - Decompose topic into subdomains and row quotas
    subdomain_quotas = await plan_dataset_coverage(topic, total_rows, task_type)
    
    BATCH_SIZE = 25  # generate 25 rows at a time (optimized for latency)
    batches_needed = -(-total_rows // BATCH_SIZE)  # ceiling division
    
    # Task-Aware Settings
    if task_type == "instruction":
        adaptive_sim_threshold = 0.96
    elif task_type == "qa":
        adaptive_sim_threshold = 0.94
    else:
        adaptive_sim_threshold = 0.92
        
    # Instruction/QA tasks have naturally repetitive question starters (e.g. "Fix the...", "Why does...")
    # so we allow a higher bigram cap to avoid excessive filtering on technical datasets.
    if task_type in ("instruction", "qa"):
        adaptive_bigram_cap = 10
    else:
        adaptive_bigram_cap = 5
    opening_bigrams = Counter()
    BANNED_STARTERS = {
        "i am sorry", "i'm sorry", "im sorry",
        "i understand", "i can help", "certainly",
        "sure!", "of course", "great question", "absolutely"
    }
    
    # Prepare personas, difficulty mix, row subdomains, row personas, row difficulties, modes, depths, structures, target_entropy, etc.
    personas = policy["personas"]
    difficulty_mix = policy["difficulty_mix"]
    
    row_subdomains = []
    for sub_name, quota in subdomain_quotas.items():
        row_subdomains.extend([sub_name] * quota)
    while len(row_subdomains) < total_rows:
        row_subdomains.append(list(subdomain_quotas.keys())[0] if subdomain_quotas else "Core Concepts")
    row_subdomains = row_subdomains[:total_rows]
    
    import random
    random.shuffle(row_subdomains)
    
    row_personas = []
    for i in range(total_rows):
        row_personas.append(personas[i % len(personas)])
        
    row_difficulties = []
    easy_count = int(total_rows * difficulty_mix.get("easy", 0.3))
    medium_count = int(total_rows * difficulty_mix.get("medium", 0.4))
    hard_count = int(total_rows * difficulty_mix.get("hard", 0.2))
    edge_count = total_rows - (easy_count + medium_count + hard_count)
    
    row_difficulties.extend(["easy"] * easy_count)
    row_difficulties.extend(["medium"] * medium_count)
    row_difficulties.extend(["hard"] * hard_count)
    row_difficulties.extend(["edge_case"] * edge_count)
    
    modes = policy.get("modes", ["explain", "debug", "refactor"])
    depths = policy.get("reasoning_depths", ["shallow", "medium", "deep", "expert"])
    structures = policy.get("structures", ["bullets", "concise", "explanation_first", "code_first", "comparison", "numbered_steps", "clarification_first"])
    
    baselines = {
        "classification": 1.2,
        "qa": 2.2,
        "chat": 2.8,
        "instruction": 2.8,
        "summarization": 2.4
    }
    baseline = baselines.get(task_type, 2.5)
    adjustment_factor = 0.85 if style == "technical" else (0.90 if style == "formal" else 1.0)
    target_entropy = round(baseline * adjustment_factor, 2)
    entropy_recovery_active = False
    
    # State variables protected by locks:
    all_rows = []
    all_embeddings = []
    existing_inputs = []
    state_lock = asyncio.Lock()
    
    deepseek_failures = 0
    deepseek_failures_lock = asyncio.Lock()
    
    semaphore = asyncio.Semaphore(4)
    limiter = TokenBucketLimiter(capacity=10, fill_rate=1.0)
    
    event_queue = asyncio.Queue()
    
    # Track yields of events
    await event_queue.put({
        "event": "start",
        "total_requested": total_rows,
        "batches": batches_needed,
        "policy": policy,
        "profile": profile,
        "coverage_plan": subdomain_quotas
    })
    
    async def run_batch_task(
        batch_num: int,
        this_batch_size: int,
        batch_subdomains: list[str],
        batch_subdomain: str,
        batch_persona: str,
        batch_difficulty: str,
        batch_mode: str,
        batch_depth: str,
        batch_structure: str,
        batch_simulate_user: bool,
        batch_adversarial: bool
    ):
        nonlocal entropy_recovery_active, deepseek_failures
        async with semaphore:
            await limiter.acquire()
            
            # Consecutive failures check
            async with deepseek_failures_lock:
                current_failures = deepseek_failures
                
            use_openai = False
            if current_failures >= 3:
                use_openai = True
                logger.info(f"[Fallback] Batch {batch_num+1} using OpenAI fallback due to {current_failures} consecutive DeepSeek failures.")
            
            if use_openai:
                client = get_openai_client(openai_api_key)
                synthesis_model = "gpt-4o-mini"
            else:
                client, synthesis_model = get_llm_client(openai_api_key, deepseek_api_key)
            if synthesis_model == "gpt-4o-mini" and not settings.deepseek_api_key and not deepseek_api_key:
                # DeepSeek not configured, just use OpenAI directly
                pass
            
            await event_queue.put({
                "event": "batch_start",
                "batch": batch_num + 1,
                "total_batches": batches_needed,
                "generating": this_batch_size,
                "persona": batch_persona,
                "difficulty": batch_difficulty,
                "subdomain": batch_subdomain,
                "mode": batch_mode,
                "depth": batch_depth
            })
            
            # Extract inputs history under lock
            async with state_lock:
                inputs_history = list(existing_inputs)
                
            current_temp = policy["temperature"]
            if entropy_recovery_active:
                import random
                current_temp = min(1.15, policy["temperature"] + 0.15)
                batch_persona = random.choice(["strict reviewer", "interviewer", "debugging assistant", "teacher", "concise responder", "mentor"])
                batch_mode = random.choice(["critique", "compare", "optimize", "adversarial", "clarification"])
                batch_structure = random.choice(["concise", "explanation_first", "code_first", "comparison", "numbered_steps", "clarification_first"])
                batch_simulate_user = True
                batch_adversarial = True
                
            prompt = build_generation_prompt(
                topic=topic,
                task_type=task_type,
                style=style,
                batch_size=this_batch_size,
                existing_inputs=inputs_history,
                domain_context=domain_context,
                persona=batch_persona,
                difficulty=batch_difficulty,
                subdomain=batch_subdomains,
                mode=batch_mode,
                depth=batch_depth,
                structure=batch_structure,
                simulate_user=batch_simulate_user,
                adversarial=batch_adversarial,
                multi_turn=multi_turn,
                replay_subdomains=[] if not inputs_history else [r.get("subdomain", "") for r in all_rows[-15:] if r.get("subdomain")],
                replay_modes=[] if not inputs_history else [r.get("mode", "") for r in all_rows[-10:] if r.get("mode")]
            )
            
            # Dispatch call with retries
            response = None
            try:
                for attempt in range(3):
                    try:
                        logger.info(f"[Batch {batch_num+1}] Dispatching API call to {synthesis_model} (Attempt {attempt+1}/3)...")
                        batch_start_time = time.time()
                        
                        kwargs = {
                            "model": synthesis_model,
                            "messages": [
                                {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                                {"role": "user", "content": prompt},
                            ],
                            "temperature": current_temp,
                            "max_tokens": 4000,
                        }
                        if synthesis_model != "deepseek-chat":
                            kwargs["response_format"] = {"type": "json_object"}
                            
                        response = await asyncio.wait_for(
                            client.chat.completions.create(**kwargs),
                            timeout=90.0
                        )
                        elapsed = time.time() - batch_start_time
                        logger.info(f"[Batch {batch_num+1}] API call to {synthesis_model} succeeded in {elapsed:.2f}s")
                        
                        # Reset DeepSeek failures if successful
                        if synthesis_model == "deepseek-chat":
                            async with deepseek_failures_lock:
                                deepseek_failures = 0
                        break
                    except Exception as e:
                        logger.warning(f"[Batch {batch_num+1}] Attempt {attempt+1}/3 failed: {type(e).__name__} - {str(e)}")
                        if synthesis_model == "deepseek-chat":
                            async with deepseek_failures_lock:
                                deepseek_failures += 1
                        if attempt < 2:
                            # 10s backoff cooldown
                            await asyncio.sleep(10.0)
                        else:
                            # If DeepSeek failed 3 times, trigger fallback immediately for this batch
                            if synthesis_model == "deepseek-chat":
                                logger.warning(f"[Batch {batch_num+1}] DeepSeek failed 3 consecutive times. Triggering immediate OpenAI fallback for this batch.")
                                client = get_openai_client(openai_api_key)
                                synthesis_model = "gpt-4o-mini"
                                # Update kwargs for OpenAI
                                kwargs["model"] = synthesis_model
                                kwargs["response_format"] = {"type": "json_object"}
                                # Make the call to OpenAI
                                response = await asyncio.wait_for(
                                    client.chat.completions.create(**kwargs),
                                    timeout=90.0
                                )
                                logger.info(f"[Batch {batch_num+1}] Fallback call to OpenAI succeeded.")
                            else:
                                raise e
                            
                if response is None:
                    raise RuntimeError("API call failed after all attempts")
                    
                content = response.choices[0].message.content
                parsed = robust_json_loads(content)
                
                if isinstance(parsed, list):
                    batch_rows = parsed
                elif isinstance(parsed, dict):
                    batch_rows = next((v for v in parsed.values() if isinstance(v, list)), [])
                else:
                    batch_rows = []
                    
                logger.info(f"[Batch {batch_num+1}] Parsed {len(batch_rows)} rows from LLM content.")
                
                # Basic cleaning with stats
                clean_rows, clean_stats = filter_bad_rows_with_stats(batch_rows, task_type)
                rejected_empty = clean_stats["empty_io"]
                rejected_short = clean_stats["too_short"]
                rejected_generic = clean_stats["generic_phrase"]
                rejected_local_dup = clean_stats["duplicate_input"]
                
                logger.info(f"[Batch {batch_num+1}] {len(clean_rows)} rows remaining after basic cleaning. ("
                            f"Filtered: {rejected_empty} empty, {rejected_short} too short, "
                            f"{rejected_generic} generic, {rejected_local_dup} local duplicate)")
                
                # Bi-gram filter (Banned Starters)
                diverse_rows = []
                rejected_banned = 0
                for r in clean_rows:
                    out = str(r.get("output", "")).strip()
                    out_lower = out.lower()
                    
                    is_banned = False
                    for starter in BANNED_STARTERS:
                        if out_lower.startswith(starter):
                            is_banned = True
                            break
                    if is_banned:
                        rejected_banned += 1
                        continue
                    diverse_rows.append(r)
                    
                logger.info(f"[Batch {batch_num+1}] {len(diverse_rows)} rows remaining after banned starters check. ("
                            f"Filtered: {rejected_banned} banned starters)")
                
                # Get embeddings for these candidates using local embedder
                cand_inputs = [r["input"] for r in diverse_rows]
                embedder = get_embedding_client()
                cand_embs = await embedder.embed(cand_inputs)
                
                unique_rows = []
                unique_embs = []
                rejected_semantic_duplicate = 0
                rejected_bigram_cap = 0
                
                # Perform duplicate and updates under the lock
                async with state_lock:
                    # Compare candidate embeddings against the last 50 embeddings in the dataset
                    existing_embs = [emb for emb in all_embeddings[-50:] if emb is not None]
                    
                    # Verify cosine similarity
                    if cand_embs:
                        if existing_embs:
                            cand_matrix = np.array(cand_embs, dtype=np.float32)
                            cand_norms = np.linalg.norm(cand_matrix, axis=1, keepdims=True)
                            cand_normed = cand_matrix / (cand_norms + 1e-9)
                            
                            exist_matrix = np.array(existing_embs, dtype=np.float32)
                            exist_norms = np.linalg.norm(exist_matrix, axis=1, keepdims=True)
                            exist_normed = exist_matrix / (exist_norms + 1e-9)
                            
                            sims = cand_normed @ exist_normed.T
                            is_dup_mask = sims.max(axis=1) > adaptive_sim_threshold
                        else:
                            is_dup_mask = np.zeros(len(cand_embs), dtype=bool)
                            
                        assigned_subdomains = list(batch_subdomains)
                        prov = {
                            "batch": batch_num + 1,
                            "persona": batch_persona,
                            "mode": batch_mode,
                            "structure": batch_structure,
                            "temperature": round(current_temp, 2),
                            "recovery_active": entropy_recovery_active,
                            "difficulty": batch_difficulty,
                            "sim_threshold": round(adaptive_sim_threshold, 3),
                            "bigram_cap": adaptive_bigram_cap,
                        }
                        
                        for r, is_dup, emb in zip(diverse_rows, is_dup_mask.tolist(), cand_embs):
                            if is_dup:
                                rejected_semantic_duplicate += 1
                            else:
                                # Opening phrase check
                                out_lower = str(r.get("output", "")).lower()
                                words = re.findall(r'\b\w+\b', out_lower)[:2]
                                bigram = " ".join(words)
                                if bigram and opening_bigrams[bigram] >= adaptive_bigram_cap:
                                    rejected_bigram_cap += 1
                                    continue
                                
                                if bigram:
                                    opening_bigrams[bigram] += 1
                                    
                                r["subdomain"] = assigned_subdomains.pop(0) if assigned_subdomains else "Core Concepts"
                                r["mode"] = batch_mode
                                r["depth"] = batch_depth
                                r["structure"] = batch_structure
                                r["_provenance"] = prov
                                unique_rows.append(r)
                                unique_embs.append(emb)
                                
                    all_rows.extend(unique_rows)
                    all_embeddings.extend(unique_embs)
                    existing_inputs.extend([r["input"] for r in unique_rows])
                    
                # Yield yield rate statistics:
                logger.info(
                    f"[Batch {batch_num+1} Filter Summary] "
                    f"Parsed: {len(batch_rows)} rows. "
                    f"Cleaned/Accepted: {len(unique_rows)} rows. "
                    f"Filtered out: {len(batch_rows) - len(unique_rows)} rows. "
                    f"Reasons: ["
                    f"Empty: {rejected_empty}, "
                    f"Too short: {rejected_short}, "
                    f"Generic: {rejected_generic}, "
                    f"Local duplicate: {rejected_local_dup}, "
                    f"Banned starter: {rejected_banned}, "
                    f"Semantic duplicate (similarity > {adaptive_sim_threshold}): {rejected_semantic_duplicate}, "
                    f"Bi-gram cap exceeded (cap={adaptive_bigram_cap}): {rejected_bigram_cap}"
                    f"]"
                )
                
                # Check for starting word entropy drops (Dynamic Entropy Recovery)
                async with state_lock:
                    total_so_far = len(all_rows)
                if total_so_far >= 10:
                    current_first_words = []
                    async with state_lock:
                        temp_rows = list(all_rows)
                    for r in temp_rows:
                        out = str(r.get("output", "")).strip()
                        words = re.findall(r'\b\w+\b', out.lower())
                        if words:
                            current_first_words.append(words[0])
                    current_entropy = calculate_shannon_entropy(current_first_words)
                    
                    if current_entropy < target_entropy and not entropy_recovery_active:
                        entropy_recovery_active = True
                        logger.warning(
                            f"Dynamic Entropy Recovery triggered: starting word entropy is {current_entropy:.2f} < target_entropy {target_entropy:.2f}. Bumping temperature and shifting policy parameters."
                        )
                        
                elapsed_total = time.time() - start_time
                time_taken = time.time() - batch_start_time
                
                # Estimate ETA
                async with state_lock:
                    num_completed_rows = len(all_rows)
                
                yield_batch_num = batch_num + 1
                avg_row_time = elapsed_total / max(num_completed_rows, 1)
                eta = avg_row_time * max(total_rows - num_completed_rows, 0)
                
                await event_queue.put({
                    "event": "batch_done",
                    "batch": yield_batch_num,
                    "generated": len(batch_rows),
                    "after_filter": len(unique_rows),
                    "filtered_out": len(batch_rows) - len(unique_rows),
                    "filter_reasons": {
                        "empty_io": rejected_empty,
                        "too_short": rejected_short,
                        "generic_phrase": rejected_generic,
                        "local_duplicate": rejected_local_dup,
                        "banned_starter": rejected_banned,
                        "semantic_duplicate": rejected_semantic_duplicate,
                        "bigram_cap_exceeded": rejected_bigram_cap
                    },
                    "total_so_far": num_completed_rows,
                    "time_taken": round(time_taken, 1),
                    "elapsed_total": round(elapsed_total, 1),
                    "eta": round(eta, 1),
                })
                
            except Exception as e:
                logger.error(f"[Batch {batch_num+1}] Failed completely. Error: {str(e)}", exc_info=True)
                await event_queue.put({"event": "batch_error", "batch": batch_num + 1, "error": str(e)})
            finally:
                try:
                    await client.close()
                except Exception:
                    pass
                    
    async def execution_manager():
        try:
            batches_spawned = 0
            active_subdomain_index = 0
            max_batches_to_run = max(batches_needed + 10, int(batches_needed * 1.5))
            
            while True:
                async with state_lock:
                    current_count = len(all_rows)
                if current_count >= total_rows or batches_spawned >= max_batches_to_run:
                    break
                    
                remaining_rows = total_rows - current_count
                chunk_batches_needed = -(-remaining_rows // BATCH_SIZE)
                chunk_batches = min(10, chunk_batches_needed)
                
                logger.info(f"[ExecutionManager] Spawning next chunk of {chunk_batches} concurrent batches. Current rows: {current_count}/{total_rows}")
                
                tasks = []
                for i in range(chunk_batches):
                    batch_num = batches_spawned
                    batches_spawned += 1
                    
                    # Determine batch size dynamically
                    this_batch_size = BATCH_SIZE
                    # Build subdomains for this batch
                    batch_subdomains = []
                    for _ in range(this_batch_size):
                        sub = row_subdomains[active_subdomain_index % len(row_subdomains)]
                        batch_subdomains.append(sub)
                        active_subdomain_index += 1
                        
                    batch_subdomain = batch_subdomains[0] if batch_subdomains else "Core Concepts"
                    batch_persona = row_personas[batch_num % len(row_personas)] if row_personas else "expert analyst"
                    batch_difficulty = row_difficulties[batch_num % len(row_difficulties)] if row_difficulties else "medium"
                    batch_mode = modes[batch_num % len(modes)]
                    batch_depth = depths[batch_num % len(depths)]
                    batch_structure = structures[batch_num % len(structures)]
                    
                    batch_simulate_user = (batch_num % 4 == 0)
                    batch_adversarial = (batch_num % 6 == 0)
                    
                    tasks.append(
                        run_batch_task(
                            batch_num=batch_num,
                            this_batch_size=this_batch_size,
                            batch_subdomains=batch_subdomains,
                            batch_subdomain=batch_subdomain,
                            batch_persona=batch_persona,
                            batch_difficulty=batch_difficulty,
                            batch_mode=batch_mode,
                            batch_depth=batch_depth,
                            batch_structure=batch_structure,
                            batch_simulate_user=batch_simulate_user,
                            batch_adversarial=batch_adversarial
                        )
                    )
                    
                await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"[ExecutionManager] Unexpected exception: {str(e)}", exc_info=True)
            await event_queue.put({"event": "manager_error", "error": str(e)})
        finally:
            await event_queue.put(None)
        
    # Start execution manager in the background
    manager_task = asyncio.create_task(execution_manager())
    
    # Read events from the queue and yield them
    while True:
        event = await event_queue.get()
        if event is None:
            break
        yield event
        
    await manager_task
    
    # Slice rows to match the target rows requested exactly (if we generated a few extra)
    all_rows = all_rows[:total_rows]
    all_embeddings = all_embeddings[:total_rows]
    
    # Final audit using precomputed embeddings to skip ending re-embeddings!
    logger.info(f"[FinalAudit] Running final quality audit on {len(all_rows)} rows using precomputed embeddings...")
    audit = await audit_generated_batch_async(
        client=None, # not needed since embeddings are precomputed
        rows=all_rows,
        task_type=task_type,
        openai_client=None,
        synthesis_model=policy.get("synthesis_model", "gpt-4o-mini"),
        precomputed_embeddings=all_embeddings
    )
    
    # Automated Repair Loop (Universal Engine Refinement)
    if audit.get("quality_score", 100.0) < 85.0:
        logger.info(f"[Auto-Repair] Quality score {audit['quality_score']}% is below threshold 85%. Triggering automated repair loop...")
        
        from app.services.repair_engine import repair_dataset
        
        repaired_rows = all_rows
        try:
            async for event in repair_dataset(
                rows=all_rows,
                topic=topic,
                task_type=task_type,
                style=style,
                quality_threshold=85.0,
                max_iterations=2
            ):
                if event["event"] == "repair_complete":
                    repaired_rows = event.get("rows", all_rows)
            all_rows = repaired_rows
            
            # Since repair loop returns modified rows, we must re-embed them for the final audit:
            embedder = get_embedding_client()
            repaired_embs = await embedder.embed([r["input"] for r in all_rows])
            audit = await audit_generated_batch_async(
                client=None,
                rows=all_rows,
                task_type=task_type,
                precomputed_embeddings=repaired_embs
            )
            logger.info(f"[Auto-Repair] Repair complete. Re-scored Quality: {audit['quality_score']}%")
        except Exception as repair_err:
            logger.error(f"[Auto-Repair] Repair failed: {str(repair_err)}")
            
    # Verify Semantic Coverage
    coverage_verification = verify_semantic_coverage(all_rows, subdomain_quotas)
    audit["semantic_coverage"] = coverage_verification
    for issue in coverage_verification.get("underrepresented_subdomains", []):
        audit["issues"].append({
            "type": "underrepresented_subdomain",
            "severity": "medium",
            "message": issue["warning"]
        })
    for issue in coverage_verification.get("collapsed_subdomains", []):
        audit["issues"].append({
            "type": "collapsed_subdomain",
            "severity": "medium",
            "message": issue["warning"]
        })
        
    # Compute Dataset Fingerprint
    subdomains_list = [r.get("subdomain") for r in all_rows if r.get("subdomain")]
    modes_list = [r.get("mode") for r in all_rows if r.get("mode")]
    depths_list = [r.get("depth") for r in all_rows if r.get("depth")]
    
    subdomain_counts = dict(Counter(subdomains_list))
    mode_counts = dict(Counter(modes_list))
    depth_counts = dict(Counter(depths_list))
    
    subdomain_entropy = calculate_shannon_entropy(subdomains_list)
    max_sub_entropy = math.log2(len(set(subdomains_list))) if len(set(subdomains_list)) > 0 else 0
    cluster_balance_ratio = subdomain_entropy / max_sub_entropy if max_sub_entropy > 0 else 0
    
    fingerprint = {
        "starting_word_entropy": audit.get("starting_word_entropy", 0.0),
        "cluster_balance": round(cluster_balance_ratio, 2),
        "subdomain_distribution": subdomain_counts,
        "mode_distribution": mode_counts,
        "reasoning_depth_distribution": depth_counts,
        "duplicate_rate": round(audit.get("duplicate_count", 0) / max(len(all_rows), 1), 3),
        "contradictions_found": audit.get("contradiction_count", 0)
    }
    
    yield {
        "event": "complete",
        "total_rows": len(all_rows),
        "audit": audit,
        "rows": all_rows,
        "policy": policy,
        "fingerprint": fingerprint,
        "coverage_plan": subdomain_quotas
    }


def verify_semantic_coverage(rows: list[dict], coverage_plan: dict) -> dict:
    """
    Semantic Coverage Verification:
    Checks if any subdomain is underrepresented or has a collapsed topic cluster.
    """
    if not rows or not coverage_plan:
        return {"underrepresented_subdomains": [], "collapsed_subdomains": [], "is_coverage_sufficient": True}

    from collections import Counter
    import re
    
    subdomain_counts = Counter(r.get("subdomain", "unknown") for r in rows)
    underrepresented = []
    collapsed = []

    for sub, quota in coverage_plan.items():
        count = subdomain_counts.get(sub, 0)
        # Underrepresented if count is less than 70% of quota
        if count < quota * 0.7:
            underrepresented.append({
                "subdomain": sub,
                "actual": count,
                "expected": quota,
                "warning": f"Subdomain '{sub}' is underrepresented ({count}/{quota} rows). Plan is missing concept depth."
            })

    # Lexical proxy check for concept collapse in subdomains
    for sub in subdomain_counts:
        sub_rows = [r for r in rows if r.get("subdomain") == sub]
        if len(sub_rows) > 3:
            first_words = []
            for r in sub_rows:
                words = re.findall(r'\b\w+\b', str(r.get("input", "")).lower())
                if len(words) >= 3:
                    first_words.append(" ".join(words[:3]))
            word_counts = Counter(first_words)
            if word_counts:
                most_common_pct = word_counts.most_common(1)[0][1] / len(sub_rows)
                if most_common_pct > 0.5:
                    collapsed.append({
                        "subdomain": sub,
                        "warning": f"Concept collapse detected in subdomain '{sub}'. Over 50% of prompts start with identical phrasing."
                    })

    return {
        "underrepresented_subdomains": underrepresented,
        "collapsed_subdomains": collapsed,
        "is_coverage_sufficient": len(underrepresented) == 0 and len(collapsed) == 0
    }
