from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from coding_interview_trainer.agent.llm import MockLLMBackend
from coding_interview_trainer.agent.service import InterviewAgent
from coding_interview_trainer.schemas import AssistanceMode, InterviewRequest


EXPECTED_SECTIONS = {
    AssistanceMode.BUG_DIAGNOSIS: {"Diagnosis", "Likely Bug Location", "Why It Fails"},
    AssistanceMode.PROGRESSIVE_HINT: {"Hint", "Why This Helps", "Next Step"},
    AssistanceMode.COMPLEXITY_EXPLANATION: {"Time Complexity", "Space Complexity"},
    AssistanceMode.FULL_SOLUTION: {"Algorithm Idea", "Code", "Complexity"},
    AssistanceMode.INTERVIEW_SIMULATION: {"Follow-up Questions", "Testing Prompt"},
}


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
    section_hits = 0
    rule_hits = 0
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
        expected = EXPECTED_SECTIONS[request.mode]
        if expected.issubset(response.sections):
            section_hits += 1

        response_lower = response.raw_response.lower()
        hint_or_diagnosis = request.mode in {
            AssistanceMode.PROGRESSIVE_HINT,
            AssistanceMode.BUG_DIAGNOSIS,
        }
        leaks_code = "```" in response.raw_response or "def " in response_lower
        if not hint_or_diagnosis or request.explicit_code_request or not leaks_code:
            rule_hits += 1

    denominator = valid_rows or 1
    return {
        "rows": len(rows),
        "valid_rows": valid_rows,
        "section_format_accuracy": section_hits / denominator,
        "mode_rule_compliance": rule_hits / denominator,
        "average_latency_ms": total_latency / denominator,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JSONL examples in mock mode.")
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("data/seed_examples.jsonl"))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.dataset), indent=2))


if __name__ == "__main__":
    main()
