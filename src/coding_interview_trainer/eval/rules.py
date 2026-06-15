from __future__ import annotations

from dataclasses import dataclass
import re

from coding_interview_trainer.schemas import AssistanceMode, InterviewRequest


EXPECTED_SECTIONS = {
    AssistanceMode.BUG_DIAGNOSIS: {
        "Diagnosis",
        "Likely Bug Location",
        "Why It Fails",
        "Failing Test Walkthrough",
        "Suggested Fix",
        "Complexity",
    },
    AssistanceMode.PROGRESSIVE_HINT: {"Hint", "Why This Helps", "Next Step"},
    AssistanceMode.COMPLEXITY_EXPLANATION: {
        "Time Complexity",
        "Space Complexity",
        "Variable Meaning",
        "Cost Drivers",
        "Corrections",
    },
    AssistanceMode.FULL_SOLUTION: {
        "Algorithm Idea",
        "Code",
        "Explanation",
        "Complexity",
        "Edge Cases",
    },
    AssistanceMode.INTERVIEW_SIMULATION: {
        "Interviewer Feedback",
        "Follow-up Questions",
        "Edge Case Challenges",
        "Complexity Prompt",
        "Testing Prompt",
    },
}

CODE_PATTERNS = (
    re.compile(r"```"),
    re.compile(r"^\s*(def|class|function|public|private|const|let|var)\s+", re.MULTILINE),
    re.compile(r"^\s*(for|while|if)\s+.+:\s*$", re.MULTILINE),
)
COMPLEXITY_PATTERN = re.compile(r"\bO\s*\([^)]+\)", re.IGNORECASE)


@dataclass(frozen=True)
class RuleResult:
    name: str
    passed: bool
    message: str


def extract_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif current:
            sections[current].append(line)
    return {key: "\n".join(value).strip() for key, value in sections.items()}


def contains_code(text: str) -> bool:
    return any(pattern.search(text) for pattern in CODE_PATTERNS)


def check_required_sections(mode: AssistanceMode, response_text: str) -> RuleResult:
    sections = extract_sections(response_text)
    missing = sorted(EXPECTED_SECTIONS[mode] - set(sections))
    return RuleResult(
        name="required_sections",
        passed=not missing,
        message="ok" if not missing else f"missing sections: {', '.join(missing)}",
    )


def check_mode_code_policy(request: InterviewRequest, response_text: str) -> RuleResult:
    has_code = contains_code(response_text)
    if request.mode == AssistanceMode.PROGRESSIVE_HINT and has_code:
        return RuleResult("mode_code_policy", False, "hint mode must not include code")
    if (
        request.mode == AssistanceMode.BUG_DIAGNOSIS
        and has_code
        and not request.explicit_code_request
    ):
        return RuleResult(
            "mode_code_policy",
            False,
            "bug diagnosis includes code without explicit_code_request=true",
        )
    if request.mode == AssistanceMode.FULL_SOLUTION and not has_code:
        return RuleResult("mode_code_policy", False, "full solution should include code")
    return RuleResult("mode_code_policy", True, "ok")


def check_complexity_presence(request: InterviewRequest, response_text: str) -> RuleResult:
    modes_requiring_complexity = {
        AssistanceMode.BUG_DIAGNOSIS,
        AssistanceMode.COMPLEXITY_EXPLANATION,
        AssistanceMode.FULL_SOLUTION,
    }
    if request.mode not in modes_requiring_complexity:
        return RuleResult("complexity_presence", True, "not required for this mode")
    passed = bool(COMPLEXITY_PATTERN.search(response_text))
    return RuleResult(
        "complexity_presence",
        passed,
        "ok" if passed else "missing Big-O complexity notation",
    )


def check_missing_info_behavior(request: InterviewRequest, response_text: str) -> RuleResult:
    if request.attempted_solution or request.mode in {
        AssistanceMode.PROGRESSIVE_HINT,
        AssistanceMode.FULL_SOLUTION,
    }:
        return RuleResult("missing_info_behavior", True, "not applicable")
    mentions_missing = "missing" in response_text.lower() or "not provided" in response_text.lower()
    return RuleResult(
        "missing_info_behavior",
        mentions_missing,
        "ok" if mentions_missing else "should state what information is missing",
    )


def evaluate_response_rules(
    request: InterviewRequest, response_text: str
) -> list[RuleResult]:
    return [
        check_required_sections(request.mode, response_text),
        check_mode_code_policy(request, response_text),
        check_complexity_presence(request, response_text),
        check_missing_info_behavior(request, response_text),
    ]
