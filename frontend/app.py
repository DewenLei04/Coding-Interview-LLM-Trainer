import os

import requests
import streamlit as st


API_URL = os.getenv("CIT_API_URL", "http://localhost:8000")

MODE_LABELS = {
    "bug_diagnosis": "Bug Diagnosis",
    "progressive_hint": "Progressive Hint",
    "complexity_explanation": "Complexity Explanation",
    "full_solution": "Full Solution",
    "interview_simulation": "Interview Simulation",
}


st.set_page_config(page_title="Coding Interview LLM Trainer", layout="wide")
st.title("Coding Interview LLM Trainer")

with st.sidebar:
    mode = st.selectbox("Assistance mode", list(MODE_LABELS), format_func=MODE_LABELS.get)
    programming_language = st.text_input("Programming language", value="Python")
    hint_level = st.slider("Hint level", min_value=1, max_value=5, value=1)
    explicit_code_request = st.checkbox("Explicitly request code")
    st.caption(f"API: {API_URL}")

problem_title = st.text_input("Problem title", value="Longest Substring Without Repeating Characters")
problem_description = st.text_area("Problem description", height=120)

left, right = st.columns(2)
with left:
    input_format = st.text_area("Input format", height=80)
    constraints = st.text_area("Constraints", height=80)
    attempted_solution = st.text_area("Attempted solution", height=240)
with right:
    output_format = st.text_area("Output format", height=80)
    example_test_cases = st.text_area("Example test cases", height=80)
    failing_test_case = st.text_area("Failing test case", height=100)
    expected_output = st.text_input("Expected output")
    wrong_output = st.text_input("Wrong output")
    error_message = st.text_area("Error message", height=80)

user_question = st.text_area("Question or extra context", height=80)

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
        response = requests.post(f"{API_URL}/analyze", json=payload, timeout=120)
        response.raise_for_status()
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
