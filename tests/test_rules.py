from coding_interview_trainer.eval.rules import evaluate_response_rules
from coding_interview_trainer.schemas import AssistanceMode, InterviewRequest


def test_hint_mode_fails_when_response_contains_code() -> None:
    request = InterviewRequest(
        problem_title="Two Sum",
        problem_description="Find two indices.",
        programming_language="Python",
        mode=AssistanceMode.PROGRESSIVE_HINT,
    )
    response = """## Hint
Use a hash map.

```python
def two_sum(nums, target):
    return []
```

## Why This Helps
It stores complements.

## Next Step
Trace the sample."""

    results = evaluate_response_rules(request, response)

    assert any(result.name == "mode_code_policy" and not result.passed for result in results)


def test_full_solution_requires_code() -> None:
    request = InterviewRequest(
        problem_title="Valid Parentheses",
        problem_description="Validate brackets.",
        programming_language="Python",
        mode=AssistanceMode.FULL_SOLUTION,
    )
    response = """## Algorithm Idea
Use a stack.

## Code
Write a stack solution.

## Explanation
Match closing brackets to the latest opening bracket.

## Complexity
O(n) time and O(n) space.

## Edge Cases
Empty input."""

    results = evaluate_response_rules(request, response)

    assert any(result.name == "mode_code_policy" and not result.passed for result in results)
