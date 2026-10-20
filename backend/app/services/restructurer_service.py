"""
Restructurer Service
Converts raw text chunks from any source (web scrape, PDF, MCP) into
clean {input, output} fine-tuning pairs using GPT-4o-mini.

Design principles:
- Grounded only: no hallucination beyond what's in the source text
- Async + batched for throughput
- Per-call timeout to prevent stale connections
- Falls back gracefully if a chunk fails
"""
import json
import asyncio
import logging
from openai import AsyncOpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)

RESTRUCTURE_SYSTEM_PROMPT = """You are a dataset curator converting source text into fine-tuning training pairs.

RULES (follow strictly):
1. GROUNDED — Only use information explicitly present in the provided text. Do NOT invent facts.
2. DIVERSE — Generate varied question/instruction styles for the same content where possible.
3. COMPLETE — Outputs must be substantive, not truncated or generic.
4. FORMAT — Respond ONLY with a valid JSON array. No markdown, no preamble.
5. SCHEMA — Each object must have "input", "output", and "reasoning" keys. The "reasoning" key should describe your step-by-step thinking or why this question/instruction is valuable.
6. CONTEXT-FREE — Formulate questions/instructions to be independent of the source text. Do NOT include phrases like "according to the text", "mentioned in the source text", "stated in the document", or "as described above". Write the questions as if a user is asking them naturally to a knowledgeable assistant.

Task types:
- qa: Generate factual questions with accurate answers from the text
- instruction: Generate clear instructions with detailed step-by-step outputs
- chat: Generate realistic conversational exchanges grounded in the text
- summarization: Generate longer input passages with concise summaries
- classification: Generate inputs with short categorical labels"""

TASK_GUIDANCE = {
    "qa": "Generate factual question-answer pairs. Questions should be specific and answerable from the text alone.",
    "instruction": "Generate actionable instructions with numbered, step-by-step outputs derived from the text.",
    "chat": "Generate natural conversational exchanges where the assistant explains concepts from the text.",
    "summarization": "Use passages from the text as input and generate concise, accurate summaries as output.",
    "classification": "Generate short inputs that can be categorized with a concise label derived from the text.",
}

STYLE_GUIDANCE = {
    "formal": "Use professional, formal language. Complete sentences. No contractions.",
    "casual": "Use natural, conversational language. Contractions are fine.",
    "technical": "Use precise technical terminology. Assume domain expertise in the reader.",
    "simple": "Use simple, clear language. Short sentences. Accessible to non-experts.",
}


def _build_restructure_prompt(
    chunk: str,
    task_type: str,
    style: str,
    pairs_per_chunk: int,
) -> str:
    task_instr = TASK_GUIDANCE.get(task_type, TASK_GUIDANCE["qa"])
    style_instr = STYLE_GUIDANCE.get(style, STYLE_GUIDANCE["casual"])
    return f"""Generate {pairs_per_chunk} training pairs from the following source text.

TASK TYPE: {task_type} — {task_instr}
STYLE: {style_instr}

SOURCE TEXT:
\"\"\"
{chunk[:3000]}
\"\"\"

Return a JSON array of exactly {pairs_per_chunk} objects with "input", "output", and "reasoning" keys.
Only use information present in the source text above. Do not hallucinate. Do not include any meta-textual phrases (like "according to the text" or "in the source text") in the inputs. Write questions as if a user is asking them to a helpful assistant."""


async def restructure_chunks_to_pairs(
    chunks: list[str],
    task_type: str = "qa",
    style: str = "formal",
    pairs_per_chunk: int = 3,
    call_timeout: float = 60.0,
    max_concurrency: int = 15,
    on_progress = None,
) -> list[dict]:
    """
    Convert a list of raw text chunks into {input, output} fine-tuning pairs.

    Args:
        chunks: List of raw text strings (each ~500–2000 chars)
        task_type: One of qa / instruction / chat / summarization / classification
        style: One of formal / casual / technical / simple
        pairs_per_chunk: How many pairs to generate per chunk (default 3)
        call_timeout: Per-call OpenAI timeout in seconds
        max_concurrency: Maximum concurrent requests to OpenAI
        on_progress: Optional async callback to stream progress back

    Returns:
        Flat list of {input, output} dicts
    """
    if not chunks:
        return []

    import time
    start_time = time.time()

    from app.services.llm_client import get_llm_client
    client, synthesis_model = get_llm_client()
    all_pairs: list[dict] = []
    seen_inputs: set[str] = set()

    sem = asyncio.Semaphore(max_concurrency)
    progress_state = {"total_pairs": 0, "completed_chunks": 0}
    progress_lock = asyncio.Lock()

    async def process_chunk(i: int, chunk: str) -> list[dict]:
        chunk = chunk.strip()
        if len(chunk) < 80:
            return []

        prompt = _build_restructure_prompt(chunk, task_type, style, pairs_per_chunk)
        chunk_pairs = []
        reasoning_list = []

        async with sem:
            for attempt in range(2):
                try:
                    response = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=synthesis_model,
                            messages=[
                                {"role": "system", "content": RESTRUCTURE_SYSTEM_PROMPT},
                                {"role": "user", "content": prompt},
                            ],
                            temperature=0.4,
                            max_tokens=2000,
                            response_format={"type": "json_object"},
                        ),
                        timeout=call_timeout,
                    )
                    try:
                        client._client.cookies.clear()
                    except Exception:
                        pass

                    content = response.choices[0].message.content
                    parsed = json.loads(content)

                    # Handle both {"data": [...]} and [...] responses
                    if isinstance(parsed, list):
                        raw_pairs = parsed
                    elif isinstance(parsed, dict):
                        raw_pairs = next(
                            (v for v in parsed.values() if isinstance(v, list)), []
                        )
                    else:
                        raw_pairs = []

                    # Validate and clean pairs for this chunk
                    for pair in raw_pairs:
                        inp = str(pair.get("input", "")).strip()
                        out = str(pair.get("output", "")).strip()
                        reasoning = str(pair.get("reasoning", "")).strip()
                        if inp and out and len(out) > 20:
                            chunk_pairs.append({
                                "input": inp,
                                "output": out,
                                "_source_chunk": i,
                            })
                            if reasoning:
                                reasoning_list.append(reasoning)

                    logger.debug(f"[Restructurer] Chunk {i}: generated {len(chunk_pairs)} pairs")

                    # Track progress and trigger callback
                    async with progress_lock:
                        progress_state["total_pairs"] += len(chunk_pairs)
                        progress_state["completed_chunks"] += 1
                        current_total = progress_state["total_pairs"]
                        current_completed = progress_state["completed_chunks"]

                    if on_progress:
                        chunk_reasoning = reasoning_list[0] if reasoning_list else "Synthesizing input/output alignments."
                        elapsed = time.time() - start_time
                        asyncio.create_task(
                            on_progress(
                                chunk_idx=i,
                                completed_chunks=current_completed,
                                num_pairs=len(chunk_pairs),
                                total_pairs_so_far=current_total,
                                reasoning=chunk_reasoning,
                                elapsed=elapsed
                            )
                        )

                    break  # success

                except asyncio.TimeoutError:
                    if attempt == 0:
                        logger.warning(f"[Restructurer] Chunk {i} timed out (attempt 1). Retrying...")
                        continue
                    else:
                        logger.error(f"[Restructurer] Chunk {i} timed out on retry. Skipping.")
                except Exception as e:
                    logger.error(f"[Restructurer] Chunk {i} failed: {e}")
                    break  # non-timeout error — skip this chunk

        return chunk_pairs

    # Execute all chunks concurrently
    tasks = [process_chunk(i, chunk) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks)

    # Flatten results and deduplicate inputs
    for chunk_pairs in results:
        for pair in chunk_pairs:
            inp = pair["input"]
            if inp not in seen_inputs:
                seen_inputs.add(inp)
                all_pairs.append(pair)

    logger.info(f"[Restructurer] Completed: {len(all_pairs)} pairs from {len(chunks)} chunks")
    return all_pairs


TABULAR_EXTRACTION_SYSTEM_PROMPT = """You are an expert data extraction assistant. Your task is to extract structured tabular information from the provided source text based on a specified list of columns.

STRICT RULES:
1. Extract ONLY facts, entities, and rows explicitly mentioned in the source text. Do not make up, extrapolate, or hallucinate any data.
2. If a column value is not mentioned in the text for a given entity, leave it as null (None).
3. Ensure every row is internally consistent and reflects the source text accurately.
4. Ensure the output matches the requested column types (string, int, float, boolean, category).

Return ONLY a JSON object with key "rows" containing an array of objects.
Each object must have exactly the columns specified. No extra keys.
"""

def _build_tabular_extraction_prompt(chunk: str, columns: list[dict]) -> str:
    col_spec = json.dumps(columns, indent=2)
    return f"""Read the following source text and extract all entities/rows that match the specified column schema.

COLUMN SPECIFICATIONS:
{col_spec}

SOURCE TEXT:
\"\"\"
{chunk}
\"\"\"

Return a JSON object: {{"rows": [{{col1: val1, col2: val2, ...}}, ...]}}"""


async def extract_tabular_from_chunks(
    chunks: list[str],
    columns: list[dict],
    openai_api_key: str = None,
    deepseek_api_key: str = None,
    call_timeout: float = 60.0,
    max_concurrency: int = 15,
    on_progress = None,
) -> list[dict]:
    """
    Extract structured rows from raw text chunks using an LLM.
    Supports BYOK mode using openai_api_key and deepseek_api_key.
    """
    if not chunks:
        return []

    import time
    start_time = time.time()

    from app.services.llm_client import get_llm_client
    client, synthesis_model = get_llm_client(openai_api_key, deepseek_api_key)
    all_rows: list[dict] = []

    sem = asyncio.Semaphore(max_concurrency)
    completed_chunks = 0
    progress_lock = asyncio.Lock()

    async def process_chunk(i: int, chunk: str) -> list[dict]:
        nonlocal completed_chunks
        chunk = chunk.strip()
        if len(chunk) < 80:
            async with progress_lock:
                completed_chunks += 1
            return []

        prompt = _build_tabular_extraction_prompt(chunk, columns)
        chunk_rows = []

        async with sem:
            for attempt in range(2):
                try:
                    kwargs = {
                        "model": synthesis_model,
                        "messages": [
                            {"role": "system", "content": TABULAR_EXTRACTION_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.1,  # Low temperature for precise factual extraction
                        "max_tokens": 2000,
                    }
                    if synthesis_model != "deepseek-chat":
                        kwargs["response_format"] = {"type": "json_object"}

                    response = await asyncio.wait_for(
                        client.chat.completions.create(**kwargs),
                        timeout=call_timeout,
                    )
                    try:
                        client._client.cookies.clear()
                    except Exception:
                        pass

                    content = response.choices[0].message.content
                    from app.services.dataset_generator import robust_json_loads
                    parsed = robust_json_loads(content)

                    if isinstance(parsed, dict) and "rows" in parsed:
                        chunk_rows = parsed["rows"]
                    elif isinstance(parsed, list):
                        chunk_rows = parsed
                    else:
                        chunk_rows = next((v for v in parsed.values() if isinstance(v, list)), [])

                    async with progress_lock:
                        completed_chunks += 1
                        elapsed = time.time() - start_time
                        if on_progress:
                            await on_progress(
                                chunk_idx=i,
                                completed_chunks=completed_chunks,
                                num_rows=len(chunk_rows),
                                total_rows_so_far=len(all_rows) + len(chunk_rows),
                                elapsed=elapsed
                            )

                    break  # success

                except asyncio.TimeoutError:
                    if attempt == 0:
                        logger.warning(f"[TabularExtractor] Chunk {i} timed out (attempt 1). Retrying...")
                        continue
                    else:
                        logger.error(f"[TabularExtractor] Chunk {i} timed out on retry. Skipping.")
                except Exception as e:
                    logger.error(f"[TabularExtractor] Chunk {i} failed: {e}")
                    break  # non-timeout error — skip this chunk

        return chunk_rows

    # Execute all chunks concurrently
    tasks = [process_chunk(i, chunk) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks)

    # Flatten results
    for chunk_rows in results:
        all_rows.extend(chunk_rows)

    logger.info(f"[TabularExtractor] Completed: Extracted {len(all_rows)} rows from {len(chunks)} chunks")
    return all_rows

