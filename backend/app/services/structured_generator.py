"""
Structured / Tabular Dataset Generator
Generates multi-column datasets for ML tasks (regression, classification, etc.)
Completely separate from the conversational fine-tuning generator.
"""
import json
import re
import hashlib
import math
from collections import Counter
from typing import AsyncGenerator
from openai import AsyncOpenAI
import random
import logging
from app.core.config import settings
from app.services.llm_client import get_llm_client
from app.services.dataset_generator import robust_json_loads

# Initialize module logger for terminal trackability
logger = logging.getLogger(__name__)


STRUCTURED_SYSTEM_PROMPT = r"""You are an expert data scientist generating synthetic tabular datasets for ML training.

STRICT RULES:
1. Every row must be statistically realistic and internally consistent
2. Values must follow real-world mathematical distributions as requested in column distribution_prior specs:
   - "normal": continuous values following a symmetric bell curve centered at the mean of the range.
   - "log_normal": continuous positive-only values skewed right with a long tail (many low values, few extremely high values).
   - "poisson": discrete non-negative integers representing counts of independent events occurring in a fixed interval (skewed right).
   - "exponential": continuous non-negative values representing time/distance intervals between Poisson events, highly clustered near zero.
   - "uniform": values spread evenly and flatly across the specified min-max range.
   - "skewed": values systematically shifted or biased towards one side of the range.
3. Relationships between columns must be logically coherent
   Example: if age=5, job_title cannot be "Senior Engineer"
   Example: if price=50000, sqft cannot be 10000 in an expensive city
4. Include realistic outliers and edge cases (5-10% of rows)
5. Vary distributions — do not cluster all values around the mean
6. No placeholder values like "N/A", "unknown", "TBD" unless specified
7. String values must be specific and realistic, not generic like "City1", "Product A"
8. Numeric values must have realistic variance and correlation patterns
9. NO LAZY MAPPINGS — Avoid simple linear increments or identical rows. Ensure data is mathematically rich and distinct.
10. CONTEXT-FREE LABELS — Ensure category labels and column values represent real-world entities (e.g. actual cities, product categories) rather than generic placeholders (e.g., 'category_1').

Return ONLY a JSON object with key "rows" containing an array of objects.
Each object must have exactly the columns specified. No extra keys.

STRICT JSON ESCAPING RULES:
1. Double quotes inside string values MUST be escaped as \" (e.g. use \" for nested quotes, or prefer single quotes inside string values).
2. Backslashes in string values (such as regex patterns or escape sequences like \n or \t) MUST be double-escaped as \\ (e.g. write \\n or \\t or \\\\).
3. Ensure the output is a valid JSON array of objects.
"""


def infer_conditional_dependencies(columns: list[dict]) -> tuple[list[str], float]:
    """
    Auto-infers conditional dependency and interaction rules from column names/types
    with a confidence score. Returns a list of rules and the average confidence score.
    """
    rules = []
    scores = []
    col_names = {c["name"].lower() for c in columns}

    # Helper to append rule with confidence score
    def add_rule(rule: str, conf: float):
        rules.append(rule)
        scores.append(conf)

    # 1. Income vs Loan Amount
    if "income" in col_names and "loan_amount" in col_names:
        add_rule("income ↑ -> loan_amount ↑ (higher income allows larger loan amounts)", 0.95)

    # 2. Credit Score vs Default / Risk
    if "credit_score" in col_names:
        risk_cols = [c for c in col_names if "default" in c or "risk" in c or "fraud" in c]
        for rc in risk_cols:
            add_rule(f"credit_score ↓ -> {rc} ↑ (lower credit score increases default risk/probability)", 0.90)

    # 3. Experience vs Salary / Income
    if "experience" in col_names:
        sal_cols = [c for c in col_names if "salary" in c or "income" in c or "pay" in c]
        for sc in sal_cols:
            add_rule(f"experience ↑ -> {sc} ↑ (more years of experience scales salary upwards)", 0.95)

    # 4. Age vs Experience
    if "age" in col_names and "experience" in col_names:
        add_rule("age ↑ -> experience ↑ (experience must be logically smaller than age - 18, and positive)", 0.99)

    # 5. Age vs Income
    if "age" in col_names and "income" in col_names:
        add_rule("age ↑ -> income ↑ (income generally increases with age, peaking between 45 and 60)", 0.80)

    # 6. Multi-feature Interactions
    if "income" in col_names and "credit_score" in col_names:
        risk_cols = [c for c in col_names if "default" in c or "risk" in c]
        for rc in risk_cols:
            add_rule(f"low income + poor credit_score + long loan_term -> strongly increase {rc} (multi-feature interaction)", 0.85)

    avg_conf = sum(scores) / len(scores) if scores else 0.0
    return rules, avg_conf


def build_schema_prompt(
    description: str,
    columns: list[dict],
    row_count: int,
    existing_rows: list[dict],
    domain_context: str = "",
) -> str:
    col_spec = json.dumps(columns, indent=2)
    avoid = ""
    if existing_rows:
        sample = existing_rows[-5:]
        avoid = f"\nALREADY GENERATED (ensure variety, do not repeat patterns):\n{json.dumps(sample, indent=2)}\n"

    domain = f"\nDOMAIN CONTEXT: {domain_context}" if domain_context else ""

    # Auto-infer conditional dependencies
    deps, conf = infer_conditional_dependencies(columns)
    deps_section = ""
    if deps and conf > 0.75:
        logger.info(f"[Dependency Engine] Inferred {len(deps)} conditional dependencies with confidence {conf:.2f}")
        deps_section = "\nCONDITIONAL DEPENDENCIES & FEATURE INTERACTIONS:\n" + "\n".join(f"- {d}" for d in deps) + "\n"

    # Enforce class imbalance controls on prediction target
    target_col = columns[-1] if columns else None
    class_imbalance_instr = ""
    if target_col:
        name_lower = target_col["name"].lower()
        col_type = target_col.get("type", "string").lower()
        
        if col_type in ("boolean", "category", "string"):
            base_rate = None
            if "fraud" in name_lower:
                base_rate = "2%"
            elif "default" in name_lower:
                base_rate = "12%"
            elif "churn" in name_lower:
                base_rate = "15%"
            elif "click" in name_lower or "convert" in name_lower:
                base_rate = "8%"
                
            if base_rate:
                class_imbalance_instr = f"\nTARGET CLASS IMBALANCE CONTROL:\n- For the target column '{target_col['name']}', ensure a realistic class imbalance. The positive target class (e.g. 1/True/Yes) should occur in approximately {base_rate} of the rows. Real-world datasets are highly imbalanced, do NOT generate a 50/50 uniform split.\n"

    return f"""Generate exactly {row_count} rows for this dataset:

DATASET DESCRIPTION: {description}{domain}

COLUMN SPECIFICATIONS:
{col_spec}

{avoid}
{deps_section}{class_imbalance_instr}
DISTRIBUTION REQUIREMENTS:
- Vary numeric values realistically (not all near average)
- Include some edge cases and outliers
- String values must be specific real-world names/places/terms
- Ensure cross-column logical consistency

Return JSON object: {{"rows": [{{col1: val1, col2: val2, ...}}, ...]}}"""


def infer_column_spec(col: dict) -> str:
    """Generate a human-readable spec for a column."""
    parts = [f"type: {col.get('type', 'string')}"]
    if col.get('range'): parts.append(f"range: {col['range']}")
    if col.get('values'): parts.append(f"allowed values: {col['values']}")
    if col.get('description'): parts.append(col['description'])
    return ", ".join(parts)


def calculate_shannon_entropy(labels: list) -> float:
    """
    Calculate the Shannon Entropy of a list of labels/categories.
    Formula: H(X) = -sum(P(x_i) * log2(P(x_i)))
    Higher entropy = more balanced categorical spread.
    """
    if not labels:
        return 0.0
    n = len(labels)
    counts = Counter(labels)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p)
    return round(entropy, 2)


def simulate_missingness(rows: list[dict], columns: list[dict], rate: float = 0.05) -> list[dict]:
    """
    SIMULATED CORRELATED MISSINGNESS:
    Simulates real-world data collection issues.
    Instead of uniform random, missingness is conditioned on latent features:
    - If a user is 'self-employed' (under employment/job), they are 3x more likely to have missing income/tax docs.
    - If a user's age is young (< 23), they are 2x more likely to have missing credit history or credit score.
    - We avoid injecting missingness into the target column to keep the target label complete.
    """
    if rate <= 0.0 or not rows:
        return rows
    
    target_col = columns[-1]["name"] if columns else None
    non_target_cols = [c["name"] for c in columns[:-1]] if len(columns) > 1 else [c["name"] for c in columns]
    if not non_target_cols:
        return rows
        
    for row in rows:
        # Detect conditions for correlated missingness
        is_self_employed = False
        for k, v in row.items():
            if k.lower() in ("employment", "job", "work", "status") and str(v).lower() in ("self-employed", "freelancer", "self employed"):
                is_self_employed = True
                break
                
        is_young = False
        for k, v in row.items():
            if k.lower() == "age":
                try:
                    if float(v) < 23:
                        is_young = True
                except (ValueError, TypeError):
                    pass
                break

        for col in non_target_cols:
            current_rate = rate
            # Scale missingness rate conditionally
            if is_self_employed and col.lower() in ("income", "tax", "tax_returns", "salary", "document"):
                current_rate = min(0.85, rate * 3.0)  # 3x scaling
            if is_young and col.lower() in ("credit_score", "credit_history", "years_credit"):
                current_rate = min(0.85, rate * 2.0)  # 2x scaling

            if random.random() < current_rate:
                # Randomly assign either None (JSON null) or an empty string to mimic database nulls
                row[col] = None if random.random() < 0.7 else ""
    return rows


def inject_label_noise(rows: list[dict], columns: list[dict], rate: float = 0.05) -> list[dict]:
    """
    LABEL NOISE INJECTION:
    Simulates human labeling errors or measurement noise in the target column (last column).
    - For classification targets: flips the label randomly to another valid category.
    - For numeric targets: injects small Gaussian noise (centered at 0, using standard deviation of the column).
    """
    if rate <= 0.0 or not rows or not columns:
        return rows
        
    target_col = columns[-1]  # The prediction target is typically the last column
    col_name = target_col["name"]
    col_type = target_col.get("type", "string")
    
    if col_type == "category" or col_type == "string":
        # Pull all unique values currently in the dataset to select a replacement label from
        unique_labels = list(set(str(row.get(col_name)) for row in rows if row.get(col_name) is not None))
        if len(unique_labels) > 1:
            for row in rows:
                if random.random() < rate:
                    current_val = str(row.get(col_name))
                    other_labels = [l for l in unique_labels if l != current_val]
                    if other_labels:
                        row[col_name] = random.choice(other_labels)
                        
    elif col_type == "boolean":
        for row in rows:
            if random.random() < rate:
                val = row.get(col_name)
                if isinstance(val, bool):
                    row[col_name] = not val
                else:
                    # Convert string/int representations of booleans
                    val_str = str(val).lower()
                    row[col_name] = False if val_str in ("true", "1") else True
                    
    elif col_type in ("int", "float", "number"):
        # Calculate column variance to scale the noise appropriately (Gaussian noise scaled to std dev)
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(col_name, 0)))
            except (ValueError, TypeError):
                pass
        if vals:
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / len(vals)
            std_dev = (variance ** 0.5) if variance > 0 else (mean * 0.1)
            
            for row in rows:
                if random.random() < rate:
                    try:
                        orig_val = float(row[col_name])
                        # Inject random Gaussian noise (standard deviation is scaled to keep it subtle)
                        noise = random.normalvariate(0, max(std_dev * 0.08, 1e-9))
                        new_val = orig_val + noise
                        row[col_name] = int(new_val) if col_type == "int" else round(new_val, 2)
                    except (ValueError, TypeError):
                        pass
                        
    return rows


def audit_structured_batch(rows: list[dict], columns: list[dict]) -> dict:
    """
    STRUCTURAL AUDITING & QUALITY CONTROL:
    Deep-audits tabular data for:
      1. Missing columns or high density of Null/placeholder values.
      2. Low Variance in numeric features (uninformative predictors).
      3. Row-level duplicate counts.
      4. Categorical distribution collapse using Shannon Entropy.
      5. Overly linear correlations (Correlation Collapse warning).
      6. TARGET LEAKAGE: Detecting if a feature perfectly correlates with or is identical to the target.
    """
    if not rows:
        return {"quality_score": 0.0, "issues": [], "total_rows": 0, "synthetic_realism_score": 0.0}

    issues = []
    score = 100.0
    total = len(rows)
    expected_cols = {c["name"] for c in columns}
    target_col = columns[-1]["name"] if columns else None

    # 1. Missing columns
    missing_col_count = 0
    for row in rows:
        missing = expected_cols - set(row.keys())
        if missing:
            missing_col_count += 1
    if missing_col_count:
        issues.append({"type": "missing_columns", "count": missing_col_count,
            "severity": "critical", "message": f"{missing_col_count} rows missing expected columns"})
        score -= 30

    # 2. Null/empty values (we only flag if it exceeds natural simulated missingness rate)
    null_count = 0
    for row in rows:
        for col in columns:
            val = row.get(col["name"])
            if val is None or str(val).strip() in ("", "null", "N/A", "TBD", "unknown"):
                null_count += 1
    if null_count > total * len(columns) * 0.22: # Allow leeway for user-selected missingness rates
        issues.append({"type": "null_values", "count": null_count,
            "severity": "high", "message": f"Excessive null/placeholder values ({null_count} detected) — check missingness configuration."})
        score -= min(20, (null_count / (total * len(columns))) * 50)

    # 3. Low variance (numeric columns stuck near same value)
    for col in columns:
        if col.get("type") in ("int", "float", "number"):
            vals = []
            for row in rows:
                try: vals.append(float(row.get(col["name"], 0)))
                except (ValueError, TypeError): pass
            if vals and len(vals) > 5:
                mean = sum(vals) / len(vals)
                variance = sum((v - mean) ** 2 for v in vals) / len(vals)
                cv = (variance ** 0.5) / (abs(mean) + 1e-9)
                if cv < 0.05:
                    issues.append({"type": "low_variance", "count": 1,
                        "severity": "medium",
                        "message": f"Column '{col['name']}' has very low variance (cv={cv:.3f}) — model won't learn useful patterns",
                        "fix": "Increase value diversity in this column"})
                    score -= 8

    # 4. Duplicate rows
    seen = set()
    dupes = 0
    for row in rows:
        key = hashlib.md5(json.dumps(row, sort_keys=True).encode()).hexdigest()
        if key in seen: dupes += 1
        seen.add(key)
    if dupes > 0:
        issues.append({"type": "duplicate_rows", "count": dupes,
            "severity": "high", "message": f"{dupes} completely duplicate rows"})
        score -= min(20, (dupes / total) * 60)

    # 5. Generic string values
    generic_terms = {"city1", "city2", "product a", "product b", "item1", "user1",
                     "example", "test", "sample", "foo", "bar", "n/a"}
    generic_count = 0
    for row in rows:
        for v in row.values():
            if str(v).lower().strip() in generic_terms:
                generic_count += 1
    if generic_count > 0:
        issues.append({"type": "generic_values", "count": generic_count,
            "severity": "medium", "message": f"{generic_count} generic placeholder values like 'City1', 'Product A'"})
        score -= min(10, generic_count * 2)

    # 6. Categorical Entropy Checks (Shannon Entropy)
    for col in columns:
        if col.get("type") == "category":
            vals = [str(row.get(col["name"], "")).strip() for row in rows]
            entropy = calculate_shannon_entropy(vals)
            unique_vals = set(vals)
            if len(unique_vals) > 1:
                max_entropy = math.log2(len(unique_vals))
                if entropy < 0.5 or (entropy / max_entropy) < 0.35:
                    issues.append({
                        "type": "collapsed_category",
                        "count": 1,
                        "severity": "medium",
                        "message": f"Categorical column '{col['name']}' has collapsed distributions (entropy={entropy:.2f}). One value dominates."
                    })
                    score -= 10

    # 7. Correlation Predictability & Target Leakage Checks
    num_cols = [c["name"] for c in columns if c.get("type") in ("int", "float", "number")]
    leakage_detected = False
    
    if len(num_cols) >= 2:
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                c1 = num_cols[i]
                c2 = num_cols[j]
                x_vals = []
                y_vals = []
                for r in rows:
                    try:
                        x_vals.append(float(r.get(c1, 0)))
                        y_vals.append(float(r.get(c2, 0)))
                    except (ValueError, TypeError):
                        pass
                if len(x_vals) > 8:
                    mean_x = sum(x_vals) / len(x_vals)
                    mean_y = sum(y_vals) / len(y_vals)
                    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals))
                    var_x = sum((x - mean_x) ** 2 for x in x_vals)
                    var_y = sum((y - mean_y) ** 2 for y in y_vals)
                    denom = (var_x * var_y) ** 0.5
                    if denom > 1e-9:
                        r_coef = cov / denom
                        # Overly perfect linear mapping
                        if abs(r_coef) > 0.995:
                            issues.append({
                                "type": "perfect_correlation",
                                "count": 1,
                                "severity": "medium",
                                "message": f"Perfect correlation detected between '{c1}' and '{c2}' (r={r_coef:.3f}). Tabular values look overly linear/artificial."
                            })
                            score -= 5
                        
                        # TARGET LEAKAGE CHECK (Numeric correlation with target column)
                        if (c1 == target_col or c2 == target_col) and abs(r_coef) > 0.98:
                            leakage_detected = True
                            leak_feature = c1 if c2 == target_col else c2
                            issues.append({
                                "type": "target_leakage",
                                "count": 1,
                                "severity": "critical",
                                "message": f"CRITICAL: Near-perfect correlation detected between feature '{leak_feature}' and target '{target_col}' (r={r_coef:.4f}). This represents high Target Reconstruction Risk."
                            })
                            score -= 35

    # Target leakage check for identical categorical/string values
    if target_col:
        for col in columns[:-1]:
            col_name = col["name"]
            identical = True
            for r in rows:
                if str(r.get(col_name)) != str(r.get(target_col)):
                    identical = False
                    break
            if identical and len(rows) > 5:
                leakage_detected = True
                issues.append({
                    "type": "target_leakage",
                    "count": 1,
                    "severity": "critical",
                    "message": f"CRITICAL: Feature column '{col_name}' is 100% identical to target column '{target_col}'. This is target leakage."
                })
                score -= 35

            # Categorical target reconstruction risk / near-perfect mappings
            mapping_counts = {}
            col_val_totals = Counter()
            for r in rows:
                val = r.get(col_name)
                t_val = r.get(target_col)
                if val is not None and t_val is not None:
                    col_val_totals[val] += 1
                    if val not in mapping_counts:
                        mapping_counts[val] = Counter()
                    mapping_counts[val][t_val] += 1
                    
            if len(col_val_totals) > 1:
                deterministic_count = 0
                total_mapped = 0
                for val, counts in mapping_counts.items():
                    most_common_cnt = counts.most_common(1)[0][1]
                    deterministic_count += most_common_cnt
                    total_mapped += sum(counts.values())
                
                if total_mapped > 5:
                    mapping_accuracy = deterministic_count / total_mapped
                    # If mapping is near-perfect (e.g., > 0.96) and has fewer categories than rows
                    if mapping_accuracy > 0.96 and len(col_val_totals) < total * 0.8:
                        leakage_detected = True
                        issues.append({
                            "type": "target_leakage",
                            "count": 1,
                            "severity": "critical",
                            "message": f"CRITICAL: Target Reconstruction Leakage detected in feature '{col_name}' (mapping accuracy={mapping_accuracy:.2%}). This feature near-perfectly reconstructs target '{target_col}'."
                        })
                        score -= 30

    final_realism = round(max(0.0, score), 1)

    return {
        "total_rows": total,
        "issues": issues,
        "quality_score": final_realism,
        "synthetic_realism_score": final_realism,
        "null_count": null_count,
        "duplicate_count": dupes,
        "generic_value_count": generic_count,
        "target_leakage_found": leakage_detected
    }


def filter_structured_rows_with_stats(rows: list[dict], columns: list[dict]) -> tuple[list[dict], dict]:
    """Remove clearly broken or empty rows during streaming and return stats."""
    expected_cols = {c["name"] for c in columns}
    seen = set()
    clean = []
    
    stats = {
        "missing_cols": 0,
        "all_null": 0,
        "duplicate": 0
    }

    for row in rows:
        # Must have all columns
        if not expected_cols.issubset(set(row.keys())):
            stats["missing_cols"] += 1
            continue
        # No all-null rows
        vals = [str(row.get(c, "")).strip() for c in expected_cols]
        if all(v in ("", "null", "N/A", "None") for v in vals):
            stats["all_null"] += 1
            continue
        # Dedup
        key = hashlib.md5(json.dumps(row, sort_keys=True).encode()).hexdigest()
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)
        clean.append({c["name"]: row[c["name"]] for c in columns})

    return clean, stats


def filter_structured_rows(rows: list[dict], columns: list[dict]) -> list[dict]:
    """Remove clearly broken or empty rows during streaming."""
    clean, _ = filter_structured_rows_with_stats(rows, columns)
    return clean



async def generate_structured_stream(
    description: str,
    columns: list[dict],
    total_rows: int,
    domain_context: str = "",
    missingness_rate: float = 0.0,
    label_noise_rate: float = 0.0,
    openai_api_key: str = None,
    deepseek_api_key: str = None,
) -> AsyncGenerator[dict, None]:
    """Stream structured dataset generation in batches, optimized with concurrent batching and logging."""
    import asyncio
    import time
    start_time = time.time()
    client, synthesis_model = get_llm_client(openai_api_key, deepseek_api_key)
    BATCH_SIZE = 25
    all_rows = []
    batches_needed = -(-total_rows // BATCH_SIZE)

    logger.info(f"Structured Generation Started: {total_rows} rows requested, batches needed: {batches_needed}")
    yield {"event": "start", "total_requested": total_rows,
           "columns": [c["name"] for c in columns], "batches": batches_needed}

    # Parallel chunk processing to balance speed vs duplicate avoidance
    CONCURRENCY_LIMIT = 4
    for chunk_start in range(0, batches_needed, CONCURRENCY_LIMIT):
        chunk_batches = min(CONCURRENCY_LIMIT, batches_needed - chunk_start)
        logger.info(f"Processing parallel batch chunk: batches {chunk_start + 1} to {chunk_start + chunk_batches} (Concurrency: {CONCURRENCY_LIMIT})")

        async def generate_single_batch(b_idx: int):
            # Calculate row count for this specific batch
            rows_so_far = len(all_rows) + (b_idx - chunk_start) * BATCH_SIZE
            remaining = total_rows - rows_so_far
            if remaining <= 0:
                return None
            this_batch = min(BATCH_SIZE, remaining)

            logger.info(f"[Batch {b_idx + 1}/{batches_needed}] Dispatching LLM call for {this_batch} rows...")
            yield_events = [{"event": "batch_start", "batch": b_idx + 1, "total_batches": batches_needed, "generating": this_batch}]

            # Pass already generated rows to avoid repetition (only those generated before this chunk)
            prompt = build_schema_prompt(
                description=description,
                columns=columns,
                row_count=this_batch,
                existing_rows=all_rows,
                domain_context=domain_context,
            )

            try:
                kwargs = {
                    "model": synthesis_model,
                    "messages": [
                        {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.85,
                    "max_tokens": 4000,
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

                if isinstance(parsed, dict) and "rows" in parsed:
                    batch_rows = parsed["rows"]
                elif isinstance(parsed, list):
                    batch_rows = parsed
                else:
                    batch_rows = next((v for v in parsed.values() if isinstance(v, list)), [])

                clean_rows, filter_stats = filter_structured_rows_with_stats(batch_rows, columns)
                
                # Apply simulated missingness and label noise
                clean_rows = simulate_missingness(clean_rows, columns, missingness_rate)
                clean_rows = inject_label_noise(clean_rows, columns, label_noise_rate)

                # Add Provenance Tracking to each row
                for r in clean_rows:
                    r["_provenance"] = {
                        "batch": b_idx + 1,
                        "temperature": 0.85,
                        "missingness_applied": missingness_rate > 0,
                        "label_noise_applied": label_noise_rate > 0
                    }
                
                logger.info(
                    f"[Batch {b_idx + 1} Filter Summary] "
                    f"Parsed: {len(batch_rows)} rows. "
                    f"Cleaned/Accepted: {len(clean_rows)} rows. "
                    f"Filtered out: {len(batch_rows) - len(clean_rows)} rows. "
                    f"Reasons: ["
                    f"Missing columns: {filter_stats['missing_cols']}, "
                    f"All null: {filter_stats['all_null']}, "
                    f"Duplicate: {filter_stats['duplicate']}"
                    f"]"
                )

                yield_events.append({
                    "event": "batch_done",
                    "batch": b_idx + 1,
                    "generated": len(batch_rows),
                    "after_filter": len(clean_rows),
                    "filtered_out": len(batch_rows) - len(clean_rows),
                    "filter_reasons": {
                        "missing_cols": filter_stats["missing_cols"],
                        "all_null": filter_stats["all_null"],
                        "duplicate": filter_stats["duplicate"]
                    },
                    "sample": clean_rows[:2] if clean_rows else [],
                })
                return {"rows": clean_rows, "events": yield_events}

            except Exception as e:
                logger.error(f"[Batch {b_idx + 1}/{batches_needed}] API Error: {str(e)}", exc_info=True)
                yield_events.append({"event": "batch_error", "batch": b_idx + 1, "error": str(e)})
                return {"rows": [], "events": yield_events}

        # Run chunk tasks concurrently
        tasks = [generate_single_batch(chunk_start + i) for i in range(chunk_batches)]
        results = await asyncio.gather(*tasks)

        # Process results in order and yield logs
        for res in results:
            if res:
                # Yield progress events
                for ev in res["events"]:
                    if ev["event"] == "batch_done":
                        elapsed = time.time() - start_time
                        completed_batches = ev["batch"]
                        avg_time_per_batch = elapsed / completed_batches
                        remaining = batches_needed - completed_batches
                        eta = avg_time_per_batch * remaining if remaining > 0 else 0.0
                        
                        ev["time_taken"] = round(avg_time_per_batch, 1)
                        ev["elapsed_total"] = round(elapsed, 1)
                        ev["eta"] = round(eta, 1)
                        ev["total_so_far"] = len(all_rows) + ev["after_filter"]
                    yield ev
                all_rows.extend(res["rows"])

    logger.info(f"All batches processed. Running tabular quality audit on {len(all_rows)} rows...")
    audit = audit_structured_batch(all_rows, columns)
    logger.info(f"Audit Complete. Realism Score: {audit['synthetic_realism_score']}%")

    # Automated Repair Loop (Universal Engine Refinement)
    if audit.get("quality_score", 100.0) < 85.0:
        logger.info(f"[Auto-Repair] Tabular Realism Score {audit['quality_score']}% is below threshold 85%. Triggering automated repair loop...")
        
        from app.services.repair_engine import repair_structured_dataset
        
        repaired_rows = all_rows
        try:
            async for event in repair_structured_dataset(
                rows=all_rows,
                columns=columns,
                description=description,
                domain_context=domain_context,
                missingness_rate=missingness_rate,
                label_noise_rate=label_noise_rate,
                quality_threshold=85.0,
                max_iterations=2
            ):
                if event["event"] == "structured_repair_complete":
                    repaired_rows = event.get("rows", all_rows)
            
            all_rows = repaired_rows
            # Re-audit repaired rows
            audit = audit_structured_batch(all_rows, columns)
            logger.info(f"[Auto-Repair] Tabular Repair Complete. Re-scored Realism: {audit['synthetic_realism_score']}%")
        except Exception as repair_err:
            logger.error(f"[Auto-Repair] Tabular Repair failed: {str(repair_err)}")

    yield {"event": "complete", "total_rows": len(all_rows),
           "audit": audit, "rows": all_rows,
           "columns": [c["name"] for c in columns]}
