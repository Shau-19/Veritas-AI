"""
Dataset Profiling and Policy Engine
Dynamically analyzes user requests to infer generation constraints, persona mixes, and distribution policies.
"""
import json
import hashlib
import logging
from openai import AsyncOpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)

_profile_cache = {}
_coverage_cache = {}

PROFILE_SYSTEM_PROMPT = """You are an expert AI data scientist.
Analyze the user's dataset generation request and return a JSON object profiling the dataset's needs.

Fields to return:
- factuality_risk: "high" | "medium" | "low" (High if questions have concrete factual answers, e.g. medical, legal, code)
- hallucination_sensitivity: "high" | "medium" | "low" (High if false information is dangerous/unacceptable)
- reasoning_complexity: "high" | "medium" | "low" (High if answers require logical deduction, calculations, or coding)
- style_entropy_needed: "high" | "medium" | "low" (High if style diversity is critical to avoid bias)
- edge_case_ratio: float between 0.05 and 0.25 (Portion of examples that should cover outliers/unusual inputs)
- primary_personas: array of strings (Select exactly 3 relevant personas from: "expert analyst", "casual user", "support agent", "concise responder", "technical specialist", "empathetic assistant", "instruction follower")
- dataset_name: string (A crisp, descriptive name for the dataset)
"""

DEFAULT_POLICY = {
    "temperature": 0.85,
    "personas": ["casual user", "support agent", "concise responder"],
    "difficulty_mix": {"easy": 0.3, "medium": 0.4, "hard": 0.2, "edge_case": 0.1},
    "factuality_risk": "medium",
    "hallucination_sensitivity": "medium",
    "dataset_name": "Custom Dataset"
}

async def profile_dataset_request(
    topic_or_desc: str,
    domain_context: str = "",
    task_type: str = "chat"
) -> dict:
    """Analyze request via LLM to classify factuality, complexity, and personas."""
    cache_key = hashlib.md5(f"{topic_or_desc}||{domain_context}||{task_type}".encode("utf-8")).hexdigest()
    if cache_key in _profile_cache:
        logger.info(f"[Profiler] Profile cache hit for key '{cache_key}' (topic: {topic_or_desc[:35]})")
        return _profile_cache[cache_key]

    try:
        from app.services.llm_client import get_llm_client
        client, synthesis_model = get_llm_client()
        prompt = f"""DATASET GENERATION REQUEST:
Topic/Description: "{topic_or_desc}"
Domain Context: "{domain_context}"
Task Type: "{task_type}"

Analyze the request and respond in valid JSON matching the instructions."""

        response = await client.chat.completions.create(
            model=synthesis_model,
            messages=[
                {"role": "system", "content": PROFILE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        try:
            client._client.cookies.clear()
        except Exception:
            pass
        content = response.choices[0].message.content
        profile = json.loads(content)
        profile["task_type"] = task_type
        _profile_cache[cache_key] = profile
        logger.info(f"[Profiler] Profile cached successfully under key '{cache_key}'")
        return profile
    except Exception as e:
        logger.warning(f"[Profiler] Profiling failed, using fallback defaults. Error: {str(e)}")
        # Fallback to default
        fallback = {
            "factuality_risk": "high" if task_type in ("qa", "instruction") else "medium",
            "hallucination_sensitivity": "high" if task_type == "qa" else "medium",
            "reasoning_complexity": "high" if task_type == "instruction" else "medium",
            "style_entropy_needed": "medium",
            "edge_case_ratio": 0.15,
            "primary_personas": ["technical specialist", "expert analyst", "instruction follower"] if task_type in ("qa", "instruction") else ["casual user", "support agent", "empathetic assistant"],
            "dataset_name": f"{topic_or_desc[:30]} Dataset",
            "task_type": task_type
        }
        _profile_cache[cache_key] = fallback
        return fallback

async def plan_dataset_coverage(
    topic: str,
    total_rows: int,
    task_type: str
) -> dict:
    """Decompose a dataset request topic into subdomains and assign row quotas."""
    cache_key = hashlib.md5(f"{topic}||{total_rows}||{task_type}".encode("utf-8")).hexdigest()
    if cache_key in _coverage_cache:
        logger.info(f"[Profiler] Coverage plan cache hit for key '{cache_key}' (topic: {topic[:35]})")
        return _coverage_cache[cache_key]

    try:
        from app.services.llm_client import get_llm_client
        client, synthesis_model = get_llm_client()

        # Scale subdomain count with dataset size to prevent semantic collapse
        # on large narrow-topic datasets. More subdomains = more granular decomposition.
        if total_rows <= 100:
            min_sub, max_sub = 5, 7
        elif total_rows <= 300:
            min_sub, max_sub = 8, 12
        elif total_rows <= 500:
            min_sub, max_sub = 12, 16
        else:
            min_sub, max_sub = 16, 20

        prompt = f"""We are building a dataset compiler. We need to decompose this topic into {min_sub} to {max_sub} logical subdomains or categories to ensure full coverage and prevent topic collapse during generation.

TOPIC: "{topic}"
TASK TYPE: "{task_type}"
TOTAL ROWS REQUESTED: {total_rows}

Since we need {total_rows} rows, we require {min_sub} to {max_sub} distinct subdomains to avoid semantic repetition.
Be granular — split broad areas into specific concept clusters (e.g. instead of "hooks", use "useState basics", "useEffect cleanup", "custom hooks patterns").

Return a JSON object with a single key "coverage" containing an array of objects. Each object must represent a subdomain and have:
- "name": string (descriptive name of the subdomain, e.g. "useState basics", "useEffect cleanup")
- "weight": float (proportion of the dataset for this subdomain, all weights must sum to exactly 1.0)

Respond ONLY in valid JSON."""

        response = await client.chat.completions.create(
            model=synthesis_model,
            messages=[
                {"role": "system", "content": "You are a senior data scientist. Return only JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        try:
            client._client.cookies.clear()
        except Exception:
            pass
        content = response.choices[0].message.content
        data = json.loads(content)
        coverage_list = data.get("coverage", [])
        
        # Calculate quotas based on weights
        quotas = {}
        allocated = 0
        for i, item in enumerate(coverage_list):
            name = item["name"]
            weight = item.get("weight", 1.0 / len(coverage_list))
            quota = int(total_rows * weight)
            quotas[name] = quota
            allocated += quota
            
        # Distribute rounding remainder to the first subdomain
        if allocated != total_rows and coverage_list:
            first_name = coverage_list[0]["name"]
            quotas[first_name] += (total_rows - allocated)
            
        _coverage_cache[cache_key] = quotas
        logger.info(f"[Profiler] Coverage plan cached successfully under key '{cache_key}'")
        return quotas

    except Exception as e:
        logger.warning(f"[Profiler] Coverage planning failed, using fallback defaults. Error: {str(e)}")
        # Fallback subdomains — scale count with dataset size
        if task_type in ("qa", "instruction"):
            base = ["Core Concepts", "Advanced Usage", "Debugging & Errors", "Edge Cases", "Best Practices",
                    "Common Patterns", "Performance", "Security", "Testing", "Real-world Examples",
                    "Anti-patterns", "Tooling & Config"]
        else:
            base = ["General Inquiries", "Complex Scenarios", "Detailed Explanations", "Troubleshooting",
                    "Alternative Methods", "Follow-up Questions", "Edge Cases", "Clarifications",
                    "Common Mistakes", "Best Practices", "Advanced Topics", "Comparisons"]

        # Pick how many fallbacks to use based on row count
        if total_rows <= 100:
            fallbacks = base[:5]
        elif total_rows <= 300:
            fallbacks = base[:8]
        elif total_rows <= 500:
            fallbacks = base[:12]
        else:
            fallbacks = base

        quotas = {}
        each = total_rows // len(fallbacks)
        for f in fallbacks:
            quotas[f] = each
        quotas[fallbacks[0]] += (total_rows - (each * len(fallbacks)))
        _coverage_cache[cache_key] = quotas
        return quotas


def build_generation_policy(profile: dict) -> dict:
    """Map profile properties to model hyper-parameters and constraints."""
    # Temperature mapping
    # High factuality risk lowers temp, high style entropy raises it
    temp = 0.85
    fact_risk = profile.get("factuality_risk", "medium")
    style_entropy = profile.get("style_entropy_needed", "medium")
    
    if fact_risk == "high":
        temp -= 0.2
    elif fact_risk == "low":
        temp += 0.1
        
    if style_entropy == "high":
        temp += 0.15
    elif style_entropy == "low":
        temp -= 0.1
        
    # Clamp temp
    temp = max(0.4, min(1.15, temp))

    # Persona Mix
    personas = profile.get("primary_personas", DEFAULT_POLICY["personas"])
    if not personas:
        personas = DEFAULT_POLICY["personas"]

    # Difficulty Distribution mix based on complexity
    complexity = profile.get("reasoning_complexity", "medium")
    edge_ratio = profile.get("edge_case_ratio", 0.15)
    
    if complexity == "high":
        difficulty_mix = {
            "easy": 0.15,
            "medium": 0.40,
            "hard": 0.30,
            "edge_case": round(edge_ratio, 2)
        }
    elif complexity == "low":
        difficulty_mix = {
            "easy": 0.50,
            "medium": 0.35,
            "hard": 0.05,
            "edge_case": round(edge_ratio, 2)
        }
    else:
        difficulty_mix = {
            "easy": 0.25,
            "medium": 0.45,
            "hard": 0.15,
            "edge_case": round(edge_ratio, 2)
        }
    
    # Normalize distribution to sum to 1.0 (excluding edge cases or adjust easy)
    total_dist = sum(difficulty_mix.values())
    if abs(total_dist - 1.0) > 0.01:
        difficulty_mix["easy"] = round(1.0 - (difficulty_mix["medium"] + difficulty_mix["hard"] + difficulty_mix["edge_case"]), 2)

    # Compile the generation modes and structures dynamically
    # General modes/structures that apply across all text formats
    modes = ["explain", "debug", "refactor", "optimize", "compare", "edge_case", "critique"]
    depths = ["shallow", "medium", "deep", "expert"]
    
    task_type = profile.get("task_type", "chat")
    if task_type in ("qa", "instruction"):
        structures = ["concise", "explanation_first", "code_first", "comparison", "numbered_steps", "clarification_first"]
    else:
        structures = ["bullets", "concise", "explanation_first", "code_first", "comparison", "numbered_steps", "clarification_first"]

    return {
        "temperature": round(temp, 2),
        "personas": personas,
        "difficulty_mix": difficulty_mix,
        "modes": modes,
        "reasoning_depths": depths,
        "structures": structures,
        "factuality_risk": fact_risk,
        "hallucination_sensitivity": profile.get("hallucination_sensitivity", "medium"),
        "dataset_name": profile.get("dataset_name", "Custom Dataset")
    }

