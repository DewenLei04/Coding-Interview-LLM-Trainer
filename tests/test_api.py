from coding_interview_trainer.api.main import analyze
from coding_interview_trainer.schemas import AssistanceMode, InterviewRequest


def test_analyze_uses_mock_backend_by_default() -> None:
    response = analyze(
        InterviewRequest(
            problem_title="Two Sum",
            problem_description="Find two indices whose values sum to target.",
            programming_language="Python",
            mode=AssistanceMode.PROGRESSIVE_HINT,
            hint_level=1,
        )
    )

    assert response.backend == "mock"
    assert "Hint" in response.sections
