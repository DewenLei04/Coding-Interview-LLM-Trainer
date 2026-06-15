from coding_interview_trainer.prompts.templates import build_prompt
from coding_interview_trainer.schemas import AssistanceMode, InterviewRequest


def test_hint_prompt_blocks_full_implementation_code() -> None:
    request = InterviewRequest(
        problem_title="Two Sum",
        problem_description="Find two indices whose values sum to target.",
        programming_language="Python",
        mode=AssistanceMode.PROGRESSIVE_HINT,
        hint_level=2,
    )

    prompt = build_prompt(request)

    assert "Do not provide full implementation code" in prompt
    assert "Provide exactly one hint for hint level 2" in prompt


def test_bug_diagnosis_prompt_blocks_code_without_explicit_request() -> None:
    request = InterviewRequest(
        problem_title="Longest Substring",
        problem_description="Find the longest substring without repeating characters.",
        programming_language="Python",
        mode=AssistanceMode.BUG_DIAGNOSIS,
        attempted_solution="def f(s): return 0",
        failing_test_case='s = "abba"',
    )

    prompt = build_prompt(request)

    assert "Do not provide full corrected code" in prompt
    assert "## Diagnosis" in prompt
