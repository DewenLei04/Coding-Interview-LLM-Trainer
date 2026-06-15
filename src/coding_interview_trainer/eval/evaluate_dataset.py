from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from coding_interview_trainer.agent.llm import MockLLMBackend
from coding_interview_trainer.agent.service import InterviewAgent
from coding_interview_trainer.eval.rules import evaluate_response_rules
from coding_interview_trainer.schemas import InterviewRequest


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                row = json.loads(line)
                row["_line_number"] = line_number
                rows.append(row)
    return rows


def evaluate(path: Path) -> dict[str, float | int]:
    agent = InterviewAgent(MockLLMBackend())
    rows = load_jsonl(path)
    valid_rows = 0
    generated_rule_passes = 0
    generated_rule_total = 0
    ideal_rule_passes = 0
    ideal_rule_total = 0
    total_latency = 0.0

    for row in rows:
        request_payload = row.get("request", row)
        try:
            request = InterviewRequest.model_validate(request_payload)
        except ValidationError:
            continue

        valid_rows += 1
        response = agent.analyze(request)
        total_latency += response.latency_ms

        generated_results = evaluate_response_rules(request, response.raw_response)
        generated_rule_total += len(generated_results)
        generated_rule_passes += sum(result.passed for result in generated_results)

        ideal_response = row.get("ideal_response")
        if isinstance(ideal_response, str) and ideal_response.strip():
            ideal_results = evaluate_response_rules(request, ideal_response)
            ideal_rule_total += len(ideal_results)
            ideal_rule_passes += sum(result.passed for result in ideal_results)

    denominator = valid_rows or 1
    generated_denominator = generated_rule_total or 1
    ideal_denominator = ideal_rule_total or 1
    return {
        "rows": len(rows),
        "valid_rows": valid_rows,
        "generated_rule_accuracy": generated_rule_passes / generated_denominator,
        "ideal_response_rule_accuracy": ideal_rule_passes / ideal_denominator,
        "average_latency_ms": total_latency / denominator,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JSONL examples in mock mode.")
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("data/seed_examples.jsonl"))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.dataset), indent=2))


if __name__ == "__main__":
    main()
