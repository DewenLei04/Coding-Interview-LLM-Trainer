# JSONL Dataset Schema

Each line is one supervised instruction example.

```json
{
  "id": "bug-longest-substring-001",
  "request": {
    "problem_title": "Longest Substring Without Repeating Characters",
    "problem_description": "Given a string s, return the length of the longest substring without repeating characters.",
    "programming_language": "Python",
    "mode": "bug_diagnosis",
    "attempted_solution": "...",
    "failing_test_case": "s = \"abba\"",
    "expected_output": "2",
    "wrong_output": "3",
    "explicit_code_request": false,
    "hint_level": 1
  },
  "ideal_response": "Structured tutor response with markdown sections."
}
```

Required request fields are `problem_title`, `problem_description`, `programming_language`, and `mode`.
Bug diagnosis, complexity explanation, and interview simulation require `attempted_solution`.
Bug diagnosis also requires at least one of `failing_test_case`, `error_message`, or `wrong_output`.
