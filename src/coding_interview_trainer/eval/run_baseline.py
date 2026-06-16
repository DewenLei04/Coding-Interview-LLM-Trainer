from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from coding_interview_trainer.agent.service import InterviewAgent
from coding_interview_trainer.eval.evaluate_dataset import load_jsonl
from coding_interview_trainer.eval.rules import evaluate_response_rules
from coding_interview_trainer.schemas import InterviewRequest


def _json_default(value: Any) -> str:
    return str(value)


def run_baseline(dataset: Path, output_dir: Path, run_name: str | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    agent = InterviewAgent()
    rows = load_jsonl(dataset)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = agent.backend.model_name.replace("/", "__")
    run_id = run_name or f"{timestamp}_{agent.backend.backend_name}_{safe_model}"

    predictions_path = output_dir / f"{run_id}.jsonl"
    summary_path = output_dir / f"{run_id}_summary.json"

    valid_rows = 0
    invalid_rows = 0
    rule_passes = 0
    rule_total = 0
    mode_counts: Counter[str] = Counter()
    latencies: list[float] = []
    output_chars: list[int] = []
    started = perf_counter()

    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            example_id = row.get("id")
            request_payload = row.get("request", row)
            try:
                request = InterviewRequest.model_validate(request_payload)
            except ValidationError as exc:
                invalid_rows += 1
                handle.write(
                    json.dumps(
                        {
                            "id": example_id,
                            "line_number": row.get("_line_number"),
                            "status": "invalid_request",
                            "error": str(exc),
                        }
                    )
                    + "\n"
                )
                continue

            response = agent.analyze(request)
            rules = evaluate_response_rules(request, response.raw_response)
            valid_rows += 1
            mode_counts[request.mode.value] += 1
            latencies.append(response.latency_ms)
            output_chars.append(len(response.raw_response))
            rule_total += len(rules)
            rule_passes += sum(rule.passed for rule in rules)

            handle.write(
                json.dumps(
                    {
                        "id": example_id,
                        "line_number": row.get("_line_number"),
                        "status": "ok",
                        "mode": request.mode.value,
                        "backend": response.backend,
                        "model_name": response.model_name,
                        "latency_ms": response.latency_ms,
                        "prompt": response.prompt,
                        "raw_response": response.raw_response,
                        "sections": response.sections,
                        "rules": [asdict(rule) for rule in rules],
                    },
                    default=_json_default,
                )
                + "\n"
            )

    denominator = valid_rows or 1
    rule_denominator = rule_total or 1
    summary: dict[str, Any] = {
        "run_id": run_id,
        "created_at": timestamp,
        "dataset": str(dataset),
        "predictions_path": str(predictions_path),
        "backend": agent.backend.backend_name,
        "model_name": agent.backend.model_name,
        "rows": len(rows),
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "mode_counts": dict(sorted(mode_counts.items())),
        "rule_accuracy": rule_passes / rule_denominator,
        "average_latency_ms": sum(latencies) / denominator,
        "max_latency_ms": max(latencies) if latencies else 0,
        "average_output_chars": sum(output_chars) / denominator,
        "total_runtime_ms": (perf_counter() - started) * 1000,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline inference and persist outputs.")
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("data/golden_eval.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/baselines"))
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()
    summary = run_baseline(args.dataset, args.output_dir, args.run_name)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
