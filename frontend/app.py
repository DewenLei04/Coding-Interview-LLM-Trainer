import os

import requests
import streamlit as st


API_URL = os.getenv("CIT_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 180

MODE_LABELS = {
    "bug_diagnosis": "Bug Diagnosis",
    "progressive_hint": "Progressive Hint",
    "complexity_explanation": "Complexity Explanation",
    "full_solution": "Full Solution",
    "interview_simulation": "Interview Simulation",
}

SAMPLE = {
    "problem_title": "Longest Substring Without Repeating Characters",
    "problem_description": "Given a string s, return the length of the longest substring without repeating characters.",
    "programming_language": "Python",
    "input_format": "s is a string.",
    "output_format": "Return an integer.",
    "constraints": "0 <= len(s) <= 5 * 10^4",
    "example_test_cases": 's = "abcabcbb" -> 3\ns = "bbbbb" -> 1',
    "attempted_solution": """def lengthOfLongestSubstring(s):
    seen = {}
    left = 0
    best = 0
    for right, ch in enumerate(s):
        if ch in seen:
            left = seen[ch] + 1
        seen[ch] = right
        best = max(best, right - left + 1)
    return best""",
    "failing_test_case": 's = "abba"',
    "expected_output": "2",
    "wrong_output": "3",
    "error_message": "",
    "user_question": "Help me understand why this fails without giving the full rewritten solution.",
}


def api_health() -> dict | None:
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
    except requests.RequestException:
        return None
    return response.json()


def format_api_error(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        return response.text
    if isinstance(detail, list):
        messages = []
        for item in detail:
            location = ".".join(str(part) for part in item.get("loc", []))
            messages.append(f"{location}: {item.get('msg')}")
        return "\n".join(messages)
    return str(detail)


st.set_page_config(page_title="Coding Interview LLM Trainer", layout="wide")
st.title("Coding Interview LLM Trainer")

with st.sidebar:
    mode = st.selectbox("Assistance mode", list(MODE_LABELS), format_func=MODE_LABELS.get)
    programming_language = st.text_input("Programming language", value=SAMPLE["programming_language"])
    hint_level = st.slider("Hint level", min_value=1, max_value=5, value=1)
    explicit_code_request = st.checkbox("Explicitly request code")
    health = api_health()
    if health:
        st.success(f"{health['backend']} backend: {health['model_name']}")
    else:
        st.warning("API is not reachable")
    st.caption(f"API: {API_URL}")

problem_title = st.text_input("Problem title", value=SAMPLE["problem_title"])
problem_description = st.text_area(
    "Problem description",
    value=SAMPLE["problem_description"],
    height=180,
)

with st.expander("Advanced problem details", expanded=False):
    detail_left, detail_right = st.columns(2)
    with detail_left:
        input_format = st.text_area("Input format", value="", height=80)
        constraints = st.text_area("Constraints", value=SAMPLE["constraints"], height=80)
    with detail_right:
        output_format = st.text_area("Output format", value="", height=80)
        example_test_cases = st.text_area(
            "Example test cases",
            value=SAMPLE["example_test_cases"],
            height=120,
        )

left, right = st.columns(2)
with left:
    attempted_solution = st.text_area(
        "Attempted solution",
        value=SAMPLE["attempted_solution"],
        height=300,
    )
with right:
    failing_test_case = st.text_area(
        "Failing test case",
        value=SAMPLE["failing_test_case"],
        height=120,
    )
    expected_output = st.text_input("Expected output", value=SAMPLE["expected_output"])
    wrong_output = st.text_input("Wrong output", value=SAMPLE["wrong_output"])
    error_message = st.text_area("Error message", value=SAMPLE["error_message"], height=80)

user_question = st.text_area(
    "Question or extra context",
    value=SAMPLE["user_question"],
    height=80,
)

if mode in {"bug_diagnosis", "complexity_explanation", "interview_simulation"} and not attempted_solution.strip():
    st.warning(f"{MODE_LABELS[mode]} needs attempted solution code.")
if mode == "bug_diagnosis" and not (
    failing_test_case.strip() or error_message.strip() or wrong_output.strip()
):
    st.warning("Bug Diagnosis needs a failing test case, error message, or wrong output.")

if st.button("Analyze", type="primary"):
    payload = {
        "problem_title": problem_title,
        "problem_description": problem_description,
        "programming_language": programming_language,
        "mode": mode,
        "input_format": input_format or None,
        "output_format": output_format or None,
        "constraints": constraints or None,
        "example_test_cases": example_test_cases or None,
        "attempted_solution": attempted_solution or None,
        "failing_test_case": failing_test_case or None,
        "expected_output": expected_output or None,
        "wrong_output": wrong_output or None,
        "error_message": error_message or None,
        "user_question": user_question or None,
        "explicit_code_request": explicit_code_request,
        "hint_level": hint_level,
    }
    try:
        response = requests.post(
            f"{API_URL}/analyze",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if not response.ok:
            st.error(format_api_error(response))
            st.stop()
    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
    else:
        result = response.json()
        st.subheader("Feedback")
        for section, content in result.get("sections", {}).items():
            with st.expander(section, expanded=True):
                st.markdown(content)
        with st.expander("Raw response"):
            st.markdown(result["raw_response"])
        with st.expander("Prompt"):
            st.code(result["prompt"])
        st.caption(
            f"{result['backend']} backend, {result['model_name']}, "
            f"{result['latency_ms']:.1f} ms"
        )
