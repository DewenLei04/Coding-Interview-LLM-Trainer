from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from coding_interview_trainer.eval.rules import evaluate_response_rules
from coding_interview_trainer.schemas import AssistanceMode, InterviewRequest


@dataclass(frozen=True)
class DatasetIssue:
    line_number: int
    example_id: str | None
    severity: str
    message: str


def iter_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[DatasetIssue]]:
    rows: list[dict[str, Any]] = []
    issues: list[DatasetIssue] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(
                    DatasetIssue(line_number, None, "error", f"invalid JSON: {exc.msg}")
                )
                continue
            row["_line_number"] = line_number
            rows.append(row)
    return rows, issues


def validate_dataset(path: Path, minimum_rows: int = 100) -> dict[str, Any]:
    rows, issues = iter_jsonl(path)
    ids: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    valid_rows = 0
    ideal_response_rule_passes = 0
    ideal_response_rule_total = 0

    for row in rows:
        line_number = int(row["_line_number"])
        example_id = row.get("id")
        if not isinstance(example_id, str) or not example_id.strip():
            issues.append(DatasetIssue(line_number, None, "error", "missing string id"))
        else:
            ids[example_id] += 1

        if "request" not in row:
            issues.append(DatasetIssue(line_number, example_id, "error", "missing request"))
            continue
        if "ideal_response" not in row:
            issues.append(
                DatasetIssue(line_number, example_id, "error", "missing ideal_response")
            )
            continue
        if not isinstance(row["ideal_response"], str) or not row["ideal_response"].strip():
            issues.append(
                DatasetIssue(line_number, example_id, "error", "ideal_response is empty")
            )
            continue

        try:
            request = InterviewRequest.model_validate(row["request"])
        except ValidationError as exc:
            issues.append(
                DatasetIssue(line_number, example_id, "error", f"invalid request: {exc}")
            )
            continue

        valid_rows += 1
        mode_counts[request.mode.value] += 1

        rule_results = evaluate_response_rules(request, row["ideal_response"])
        ideal_response_rule_total += len(rule_results)
        for result in rule_results:
            if result.passed:
                ideal_response_rule_passes += 1
            else:
                issues.append(
                    DatasetIssue(
                        line_number,
                        example_id,
                        "warning",
                        f"ideal_response {result.name}: {result.message}",
                    )
                )

    for duplicate_id, count in ids.items():
        if count > 1:
            issues.append(
                DatasetIssue(0, duplicate_id, "error", f"duplicate id appears {count} times")
            )

    missing_modes = sorted(set(mode.value for mode in AssistanceMode) - set(mode_counts))
    for mode in missing_modes:
        issues.append(DatasetIssue(0, None, "warning", f"missing mode coverage: {mode}"))

    if valid_rows < minimum_rows:
        issues.append(
            DatasetIssue(
                0,
                None,
                "warning",
                f"dataset has {valid_rows} valid rows; target is at least {minimum_rows}",
            )
        )

    error_count = sum(issue.severity == "error" for issue in issues)
    warning_count = sum(issue.severity == "warning" for issue in issues)
    denominator = ideal_response_rule_total or 1
    return {
        "path": str(path),
        "rows": len(rows),
        "valid_rows": valid_rows,
        "mode_counts": dict(sorted(mode_counts.items())),
        "ideal_response_rule_accuracy": ideal_response_rule_passes / denominator,
        "errors": error_count,
        "warnings": warning_count,
        "issues": [asdict(issue) for issue in issues],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate JSONL dataset examples.")
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("data/seed_examples.jsonl"))
    parser.add_argument("--minimum-rows", type=int, default=100)
    args = parser.parse_args()
    report = validate_dataset(args.dataset, minimum_rows=args.minimum_rows)
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if report["errors"] else 0)


if __name__ == "__main__":
    main()
