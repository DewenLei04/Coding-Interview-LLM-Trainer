from collections.abc import Callable

from coding_interview_trainer.schemas import AssistanceMode, InterviewRequest


GLOBAL_RULES = """You are a coding interview tutor, not a generic chatbot.
Follow these rules:
- Do not hallucinate constraints not included in the problem statement.
- If information is missing, say exactly what is missing.
- Label assumptions clearly.
- Prefer concrete examples over vague advice.
- Use the failing test case when available.
- Be concise, but explain enough for a candidate to learn.
- Do not claim a solution is correct unless the reasoning supports it.
- If uncertain, say what needs to be tested.
"""


def _field(label: str, value: str | None) -> str:
    clean_value = value.strip() if isinstance(value, str) else ""
    return f"{label}: {clean_value if clean_value else '[not provided]'}"


def _context(request: InterviewRequest) -> str:
    fields = [
        _field("Problem title", request.problem_title),
        _field("Problem description", request.problem_description),
        _field("Input format", request.input_format),
        _field("Output format", request.output_format),
        _field("Constraints", request.constraints),
        _field("Example test cases", request.example_test_cases),
        _field("Programming language", request.programming_language),
        _field("Attempted solution", request.attempted_solution),
        _field("Failing test case", request.failing_test_case),
        _field("Error message", request.error_message),
        _field("Wrong output", request.wrong_output),
        _field("Expected output", request.expected_output),
        _field("User question", request.user_question),
    ]
    return "\n".join(fields)


def bug_diagnosis_prompt(request: InterviewRequest) -> str:
    code_rule = (
        "The user explicitly requested code, so a minimal patch or snippet is allowed."
        if request.explicit_code_request
        else "Do not provide full corrected code. Suggest the fix in words only."
    )
    return f"""{GLOBAL_RULES}
Mode: Bug Diagnosis
Task: Identify the most likely bug in the submitted code.
{code_rule}

Required response sections:
## Diagnosis
## Likely Bug Location
## Why It Fails
## Failing Test Walkthrough
## Suggested Fix
## Complexity

Interview context:
{_context(request)}
"""


def progressive_hint_prompt(request: InterviewRequest) -> str:
    return f"""{GLOBAL_RULES}
Mode: Progressive Hint
Task: Provide exactly one hint for hint level {request.hint_level}.
Hint levels:
1. Conceptual hint
2. Data structure hint
3. Algorithmic hint
4. Edge case hint
5. Near-solution hint

Do not provide full implementation code in Hint Mode.
Do not reveal the complete solution.

Required response sections:
## Hint
## Why This Helps
## Next Step

Interview context:
{_context(request)}
"""


def complexity_explanation_prompt(request: InterviewRequest) -> str:
    return f"""{GLOBAL_RULES}
Mode: Complexity Explanation
Task: Explain the submitted code's time and space complexity.

Required response sections:
## Time Complexity
## Space Complexity
## Variable Meaning
## Cost Drivers
## Corrections

Interview context:
{_context(request)}
"""


def full_solution_prompt(request: InterviewRequest) -> str:
    return f"""{GLOBAL_RULES}
Mode: Full Solution
Task: Provide a complete correct solution because the user explicitly selected Full Solution Mode.

Required response sections:
## Algorithm Idea
## Code
## Explanation
## Complexity
## Edge Cases

Interview context:
{_context(request)}
"""


def interview_simulation_prompt(request: InterviewRequest) -> str:
    return f"""{GLOBAL_RULES}
Mode: Interview Simulation
Task: Act like a technical interviewer. Ask follow-up questions and guide the candidate.
Avoid giving the answer too early.

Required response sections:
## Interviewer Feedback
## Follow-up Questions
## Edge Case Challenges
## Complexity Prompt
## Testing Prompt

Interview context:
{_context(request)}
"""


PROMPT_BUILDERS: dict[AssistanceMode, Callable[[InterviewRequest], str]] = {
    AssistanceMode.BUG_DIAGNOSIS: bug_diagnosis_prompt,
    AssistanceMode.PROGRESSIVE_HINT: progressive_hint_prompt,
    AssistanceMode.COMPLEXITY_EXPLANATION: complexity_explanation_prompt,
    AssistanceMode.FULL_SOLUTION: full_solution_prompt,
    AssistanceMode.INTERVIEW_SIMULATION: interview_simulation_prompt,
}


def build_prompt(request: InterviewRequest) -> str:
    return PROMPT_BUILDERS[request.mode](request)
