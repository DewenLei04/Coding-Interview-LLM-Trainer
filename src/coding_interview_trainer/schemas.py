from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class AssistanceMode(StrEnum):
    BUG_DIAGNOSIS = "bug_diagnosis"
    PROGRESSIVE_HINT = "progressive_hint"
    COMPLEXITY_EXPLANATION = "complexity_explanation"
    FULL_SOLUTION = "full_solution"
    INTERVIEW_SIMULATION = "interview_simulation"


class InterviewRequest(BaseModel):
    problem_title: str = Field(..., min_length=1)
    problem_description: str = Field(..., min_length=1)
    programming_language: str = Field(..., min_length=1)
    mode: AssistanceMode
    input_format: str | None = None
    output_format: str | None = None
    constraints: str | None = None
    example_test_cases: str | None = None
    attempted_solution: str | None = None
    failing_test_case: str | None = None
    error_message: str | None = None
    wrong_output: str | None = None
    expected_output: str | None = None
    user_question: str | None = None
    explicit_code_request: bool = False
    hint_level: int = Field(default=1, ge=1, le=5)

    @model_validator(mode="after")
    def validate_mode_inputs(self) -> "InterviewRequest":
        if self.mode in {
            AssistanceMode.BUG_DIAGNOSIS,
            AssistanceMode.COMPLEXITY_EXPLANATION,
            AssistanceMode.INTERVIEW_SIMULATION,
        } and not self.attempted_solution:
            raise ValueError("attempted_solution is required for this assistance mode")
        if self.mode == AssistanceMode.BUG_DIAGNOSIS and not (
            self.failing_test_case or self.error_message or self.wrong_output
        ):
            raise ValueError(
                "bug diagnosis requires a failing_test_case, error_message, or wrong_output"
            )
        return self


class AnalyzeResponse(BaseModel):
    mode: AssistanceMode
    model_name: str
    backend: Literal["mock", "transformers"]
    prompt: str
    raw_response: str
    sections: dict[str, str]
    latency_ms: float
