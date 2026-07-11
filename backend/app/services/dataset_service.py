import pandas as pd
import io
import json


def audit_dataset(file_content: bytes, file_format: str) -> dict:
    issues = []

    try:
        if file_format == "csv":
            df = pd.read_csv(io.BytesIO(file_content))
        elif file_format == "jsonl":
            lines = file_content.decode("utf-8").strip().split("\n")
            records = [json.loads(line) for line in lines if line.strip()]
            df = pd.DataFrame(records)
        else:
            return {"error": "Unsupported format", "quality_score": 0, "issues": []}

        row_count = len(df)
        if row_count == 0:
            return {"error": "Empty dataset", "quality_score": 0, "issues": []}

        score = 100.0

        # 1. Duplicate rows
        duplicate_count = int(df.duplicated().sum())
        if duplicate_count > 0:
            pct = duplicate_count / row_count
            issues.append({
                "type": "duplicate_rows",
                "count": duplicate_count,
                "severity": "high" if pct > 0.1 else "medium",
                "message": f"{duplicate_count} duplicate rows found ({pct:.1%})"
            })
            score -= min(30, pct * 100)

        # 2. Detect input/output columns
        input_col = next(
            (c for c in df.columns if c.lower() in ["input", "prompt", "instruction", "question"]),
            None
        )
        output_col = next(
            (c for c in df.columns if c.lower() in ["output", "completion", "response", "answer"]),
            None
        )

        avg_input_len = 0.0
        avg_output_len = 0.0

        if not input_col or not output_col:
            issues.append({
                "type": "missing_columns",
                "count": 1,
                "severity": "critical",
                "message": "Could not find input/output columns. Use: input/output, prompt/completion, or instruction/response"
            })
            score -= 40
        else:
            # 3. Empty values
            empty_inputs = int(df[input_col].isna().sum() + (df[input_col] == "").sum())
            empty_outputs = int(df[output_col].isna().sum() + (df[output_col] == "").sum())
            total_empty = empty_inputs + empty_outputs
            if total_empty > 0:
                issues.append({
                    "type": "empty_values",
                    "count": total_empty,
                    "severity": "high",
                    "message": f"{empty_inputs} empty inputs, {empty_outputs} empty outputs"
                })
                score -= min(20, (total_empty / row_count) * 50)

            # 4. Very short outputs
            short_outputs = int((df[output_col].astype(str).str.len() < 5).sum())
            if short_outputs > row_count * 0.05:
                issues.append({
                    "type": "short_outputs",
                    "count": short_outputs,
                    "severity": "medium",
                    "message": f"{short_outputs} outputs are very short (< 5 chars)"
                })
                score -= 10

            avg_input_len = float(df[input_col].astype(str).str.len().mean())
            avg_output_len = float(df[output_col].astype(str).str.len().mean())

        return {
            "row_count": row_count,
            "duplicate_rows": duplicate_count,
            "avg_input_length": round(avg_input_len, 1),
            "avg_output_length": round(avg_output_len, 1),
            "quality_score": round(max(0, score), 1),
            "issues": issues,
            "column_names": list(df.columns),
        }

    except Exception as e:
        return {
            "error": str(e),
            "quality_score": 0,
            "issues": [{"type": "parse_error", "count": 1, "severity": "critical", "message": str(e)}]
        }


def estimate_training_cost(
    row_count: int,
    base_model: str,
    epochs: int,
    gpu_price_per_hour: float = 0.34,
) -> dict:
    avg_tokens_per_row = 256
    throughput = {
        "llama": 2800,
        "mistral": 2600,
        "phi": 3500,
        "gemma": 2400,
    }
    model_family = next(
        (k for k in throughput if k in base_model.lower()), "llama"
    )
    tokens_per_sec = throughput[model_family]
    total_tokens = row_count * avg_tokens_per_row * epochs
    estimated_seconds = total_tokens / tokens_per_sec
    estimated_hours = estimated_seconds / 3600
    estimated_cost = estimated_hours * gpu_price_per_hour

    return {
        "estimated_seconds": round(estimated_seconds),
        "estimated_minutes": round(estimated_seconds / 60, 1),
        "estimated_cost_usd": round(estimated_cost, 4),
        "gpu_type": "RTX 4090",
        "gpu_price_per_hour": gpu_price_per_hour,
        "total_tokens": total_tokens,
    }