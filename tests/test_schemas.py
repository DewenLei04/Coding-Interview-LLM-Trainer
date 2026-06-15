import pytest
from pydantic import ValidationError

from coding_interview_trainer.schemas import AssistanceMode, InterviewRequest


def test_bug_diagnosis_requires_failure_signal() -> None:
    with pytest.raises(ValidationError):
        InterviewRequest(
            problem_title="Problem",
            problem_description="Description",
            programming_language="Python",
            mode=AssistanceMode.BUG_DIAGNOSIS,
            attempted_solution="def f(): pass",
        )


def test_progressive_hint_does_not_require_attempted_solution() -> None:
    request = InterviewRequest(
        problem_title="Problem",
        problem_description="Description",
        programming_language="Python",
        mode=AssistanceMode.PROGRESSIVE_HINT,
    )

    assert request.hint_level == 1
