# PRD: Coding Interview LLM Trainer

## 1. Project Overview

**Project Name:** Coding Interview LLM Trainer

**Primary Base Model:** `Qwen/Qwen2.5-Coder-7B-Instruct`

**Fallback Base Model:** `Qwen/Qwen2.5-Coder-3B-Instruct`

**Target Hardware:** Single NVIDIA RTX 5080 GPU with 16GB VRAM.

The Coding Interview LLM Trainer is a local AI-powered coding interview coach. The system helps users practice technical interview problems by analyzing attempted solutions, identifying logical bugs, providing progressive hints, explaining time and space complexity, and simulating interviewer-style follow-up questions.

This project is not intended to train a large language model from scratch. The project will fine-tune an existing open-source code language model using LoRA or QLoRA. The main objective is to demonstrate a complete AI engineering workflow:

1. Dataset construction.
2. Prompt design.
3. Baseline model evaluation.
4. LoRA/QLoRA fine-tuning.
5. Inference API deployment.
6. Frontend demo.
7. Evaluation and benchmarking.

The project should be portfolio-quality and suitable for a software engineering, AI engineering, or ML infrastructure resume.

The system should behave like a coding interview tutor, not like a generic chatbot. It should avoid immediately giving away the full answer when the user asks for a hint. It should guide the user toward the solution through structured feedback.

## 2. Model Decision

The default model for this project is:

```text
Qwen/Qwen2.5-Coder-7B-Instruct
```

This model is selected because:

1. It is designed for coding tasks.
2. It is instruction-tuned, making it suitable for tutor-style interaction.
3. It supports code generation, code reasoning, and code fixing.
4. It is small enough to be realistic for QLoRA fine-tuning on a single RTX 5080, assuming conservative training settings.
5. It has stronger project relevance than a generic chat model.
6. It has better instruction-following suitability than non-instruct code completion models.

The fallback model is:

```text
Qwen/Qwen2.5-Coder-3B-Instruct
```

The fallback model should be used if the 7B model causes out-of-memory errors during local training or inference.

Initial training should use QLoRA rather than full fine-tuning. Full fine-tuning is out of scope for the first version because of VRAM limits.

Recommended first training configuration:

```yaml
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
fallback_model: Qwen/Qwen2.5-Coder-3B-Instruct
training_method: qlora
load_in_4bit: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
max_seq_length: 2048
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 2e-4
num_train_epochs: 2
bf16: true
gradient_checkpointing: true
```

For the MVP, the system must support baseline inference before fine-tuning. Fine-tuning should be added after the data pipeline and evaluation pipeline are working.

## 3. Target Users

The target users are:

1. Computer science students preparing for coding interviews.
2. Junior developers practicing algorithms and data structures.
3. Software engineering candidates who want feedback on failed solutions.
4. AI engineering recruiters or interviewers reviewing the project as a portfolio demo.
5. The project owner, who wants to demonstrate practical LLM training, evaluation, deployment, and benchmarking skills.

## 4. Core User Workflow

The user should be able to submit:

1. Problem title.
2. Problem description.
3. Input and output format.
4. Constraints.
5. Example test cases.
6. Attempted solution code.
7. Programming language.
8. Failing test case.
9. Error message or wrong output.
10. Requested assistance mode.

The system then returns structured feedback based on the selected mode.

Example user request:

```text
Problem: Longest Substring Without Repeating Characters
Language: Python
Mode: Bug Diagnosis
My code:
...
Failing test case:
s = "abba"
Expected output: 2
My output: 3
```

Example expected assistant behavior:

```text
Diagnosis:
The main issue is that your left pointer can move backward when a repeated character was last seen before the current window.

Why it fails:
For input "abba", after processing the second "b", the left pointer moves correctly. However, when the final "a" is processed, your code may update left using an old index that is no longer inside the current window.

Suggested fix:
Only update left when last_seen[current_char] >= left.

Complexity:
The intended solution is O(n) time because each character is processed once. The space complexity is O(k), where k is the number of distinct characters stored in the hash map.
```

## 5. Assistance Modes

The application must support five assistance modes.

### 5.1 Bug Diagnosis Mode

The model should identify the most likely bug in the submitted code.

The response should include:

1. A short diagnosis.
2. The likely bug location.
3. Why the bug causes failure.
4. A walkthrough using the failing test case.
5. A suggested fix in words.
6. Code only if the user explicitly requests code.

The model should not rewrite the full solution unless the request is clearly asking for a full solution.

### 5.2 Progressive Hint Mode

The model should provide hints in stages.

The response should include one hint at a time unless the user asks for more.

Hint levels:

1. Conceptual hint.
2. Data structure hint.
3. Algorithmic hint.
4. Edge case hint.
5. Near-solution hint.

The model should not reveal the full implementation in Hint Mode.

### 5.3 Complexity Explanation Mode

The model should explain the submitted code’s time and space complexity.

The response should include:

1. Time complexity.
2. Space complexity.
3. Explanation of what variables represent.
4. Explanation of loops, recursion, data structures, or sorting costs.
5. Correction if the user’s stated complexity is wrong.

### 5.4 Full Solution Mode

The model may provide a complete correct solution only when the user explicitly selects Full Solution Mode.

The response should include:

1. Algorithm idea.
2. Correct code.
3. Explanation.
4. Complexity analysis.
5. Important edge cases.

### 5.5 Interview Simulation Mode

The model should act like a technical interviewer.

The response should include:

1. Follow-up questions.
2. Edge case challenges.
3. Complexity improvement prompts.
4. Requests for clarification.
5. Suggestions for testing the code.

The model should avoid giving the answer too early.

## 6. LLM Response Rules

The model must follow these rules:

1. Do not provide full code in Hint Mode.
2. Do not provide full code in Bug Diagnosis Mode unless explicitly requested.
3. Do not hallucinate constraints not included in the problem statement.
4. If the user input is incomplete, state what information is missing.
5. When making an assumption, label it clearly.
6. Prefer concrete examples over vague advice.
7. Use failing test cases when available.
8. Separate diagnosis, explanation, suggested fix, and complexity.
9. Be concise but sufficiently explanatory.
10. Do not claim that a solution is correct unless the reasoning supports it.
11. If uncertain, say what needs to be tested.
12. Avoid overconfident claims about runtime if the code is incomplete.

## 7. MVP Requirements

The MVP should include:

1. Project scaffold.
2. Local JSONL dataset format.
3. At least 100 instruction examples.
4. Prompt templates for all five assistance modes.
5. Baseline inference using `Qwen/Qwen2.5-Coder-7B-Instruct` or the fallback 3B model.
6. FastAPI backend.
7. Streamlit frontend.
8. Evaluation script with at least three metrics.
9. Basic inference benchmark.
10. README with setup and usage instructions.

The MVP does not need to complete fine-tuning on the first implementation pass. However, the repository must be structured so that fine-tuning can be added without major refactoring.

## 8. Post-MVP Requirements

Post-MVP improvements should include:

1. QLoRA fine-tuning.
2. Model comparison between baseline and fine-tuned model.
3. Larger instruction dataset.
4. Evaluation dashboard.
5. Multiple programming language support.
6. User session history.
7. Automatic safe code execution in a sandbox.
8. RAG over algorithm notes.
9. vLLM or llama.cpp inference backend.
10. Quantized inference comparison.
11. React or Next.js frontend.
12. Dockerized deployment.
13. GitHub Actions for linting and tests.

## 9. Dataset Design

The dataset should use JSONL format.

Main dataset files:

```text
data/raw/
data/processed/
data/train.jsonl
data/valid.jsonl
data/eval_cases.jsonl
```

Each training record should follow this structure:

```json
{
  "instruction": "Analyze the student's solution and identify the bug. Do not provide the full corrected code unless explicitly requested.",
  "input": "Problem: Longest Substring Without Repeating Characters\nLanguage: Python\nStudent code: ...\nFailing test case: s = \"abba\"\nExpected output: 2\nActual output: 3",
  "output": "The bug is that the left pointer can move backward when a repeated character was last seen before the current window. Only update the left pointer if the previous index of the current character is greater than or equal to the current left pointer."
}
```

The dataset should include examples for:

1. Bug diagnosis.
2. Progressive hints.
3. Complexity explanation.
4. Full solution explanation.
5. Interview follow-up questions.
6. Edge case identification.
7. Incorrect complexity correction.
8. Wrong data structure choice.
9. Off-by-one errors.
10. Sliding window bugs.
11. Stack usage bugs.
12. Linked list pointer bugs.
13. Binary search boundary bugs.
14. Recursion base case bugs.
15. Dynamic programming transition bugs.

Recommended MVP dataset size:

```text
Bug diagnosis examples: 40
Progressive hint examples: 25
Complexity explanation examples: 20
Full solution examples: 10
Interview simulation examples: 5
Total MVP examples: 100
```

Recommended full project dataset size:

```text
Bug diagnosis examples: 300
Progressive hint examples: 300
Complexity explanation examples: 200
Full solution examples: 100
Interview simulation examples: 100
Total full dataset: 1000
```

## 10. Evaluation Design

The evaluation system must compare:

1. Base model.
2. Base model with structured prompt.
3. Fine-tuned model.
4. Fine-tuned model with structured prompt.

Minimum MVP metrics:

1. Bug localization accuracy.
2. Full-answer leakage rate in Hint Mode.
3. Response format consistency.

Post-MVP metrics:

1. Hint usefulness score.
2. Complexity correctness.
3. Edge case identification accuracy.
4. Latency.
5. Tokens per second.
6. Peak VRAM usage.
7. Model loading time.

The evaluation results should be saved to:

```text
evaluation/reports/
benchmark/results/
```

The evaluation system should not depend entirely on another LLM judge. Deterministic checks should be implemented where possible.

Examples:

1. Leakage detector checks whether Hint Mode contains full code blocks.
2. Response format checker verifies required section headers.
3. Complexity checker compares expected complexity labels.
4. Keyword-based bug localization checks whether the model identifies the intended bug category.

## 11. Technical Stack

Core language:

```text
Python 3.10+
```

Model training:

```text
PyTorch
Hugging Face Transformers
Hugging Face Datasets
PEFT
TRL
bitsandbytes
optional Unsloth
```

Inference:

```text
Transformers pipeline
FastAPI inference service
optional vLLM
optional llama.cpp
```

Backend:

```text
FastAPI
Pydantic
Uvicorn
SQLite
```

Frontend:

```text
Streamlit for MVP
React or Next.js optional for post-MVP
```

Evaluation:

```text
custom Python scripts
pytest
CSV reports
Markdown reports
optional LLM-as-judge
```

Code quality:

```text
ruff
black
mypy optional
pytest
```

DevOps:

```text
Docker
docker-compose
.env configuration
Makefile
GitHub Actions
```

## 12. Proposed Project Structure

```text
coding-interview-llm-trainer/
├── PRD.md
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── docker-compose.yml
├── Makefile
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── eval_cases.jsonl
│
├── configs/
│   ├── train_qlora.yaml
│   ├── inference.yaml
│   └── eval.yaml
│
├── scripts/
│   ├── build_dataset.py
│   ├── clean_dataset.py
│   ├── split_dataset.py
│   └── convert_to_chat_format.py
│
├── training/
│   ├── finetune_qlora.py
│   ├── merge_adapter.py
│   ├── train_utils.py
│   └── callbacks.py
│
├── inference/
│   ├── model_loader.py
│   ├── prompt_templates.py
│   ├── generate.py
│   └── response_parser.py
│
├── evaluation/
│   ├── run_eval.py
│   ├── metrics.py
│   ├── leakage_detector.py
│   ├── complexity_checker.py
│   └── reports/
│
├── benchmark/
│   ├── benchmark_inference.py
│   ├── benchmark_memory.py
│   └── results/
│
├── app/
│   ├── api.py
│   ├── schemas.py
│   ├── database.py
│   └── streamlit_app.py
│
├── tests/
│   ├── test_prompt_templates.py
│   ├── test_leakage_detector.py
│   ├── test_metrics.py
│   └── test_api.py
│
└── docs/
    ├── dataset_design.md
    ├── training_notes.md
    └── evaluation_methodology.md
```

## 13. API Requirements

The FastAPI backend should expose the following endpoints.

### Health Check

```text
GET /health
```

Returns model status and system status.

### Analyze Code

```text
POST /analyze
```

Request body:

```json
{
  "problem_title": "Longest Substring Without Repeating Characters",
  "problem_description": "...",
  "constraints": "...",
  "examples": "...",
  "language": "Python",
  "attempted_code": "...",
  "failing_test_case": "s = \"abba\"",
  "expected_output": "2",
  "actual_output": "3",
  "mode": "bug_diagnosis"
}
```

Response body:

```json
{
  "mode": "bug_diagnosis",
  "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "response": "...",
  "latency_ms": 1234,
  "tokens_generated": 256
}
```

### Benchmark Results

```text
GET /benchmark/results
```

Returns latest benchmark summary.

## 14. Frontend Requirements

The MVP frontend should use Streamlit.

The UI should include:

1. Problem title input.
2. Problem description text area.
3. Constraints text area.
4. Examples text area.
5. Programming language selector.
6. Attempted code text area.
7. Failing test case input.
8. Expected output input.
9. Actual output input.
10. Assistance mode selector.
11. Submit button.
12. Structured response display.

The frontend does not need advanced styling for MVP. It should prioritize usability and clarity.

## 15. Benchmarking Requirements

The benchmark module should measure:

1. Model loading time.
2. Prompt length.
3. Generated token count.
4. Total latency.
5. Time to first token if available.
6. Tokens per second.
7. Peak VRAM usage.
8. Quantization mode.
9. Max sequence length tested.

Benchmark output should be saved as:

```text
benchmark/results/inference_benchmark.csv
benchmark/results/inference_benchmark.md
```

The benchmark should test at least:

1. Short prompt.
2. Medium prompt.
3. Long prompt with full problem statement and code.
4. Hint Mode.
5. Bug Diagnosis Mode.
6. Full Solution Mode.

## 16. Non-Goals

The initial project will not:

1. Train a large language model from scratch.
2. Full fine-tune a 7B model.
3. Guarantee that generated code is always correct.
4. Execute untrusted user code without a secure sandbox.
5. Depend on paid APIs for core functionality.
6. Replace professional interview coaching.
7. Support every programming language in the MVP.
8. Build a production-grade authentication system.
9. Build a polished commercial UI in the first version.

## 17. Success Criteria

The project is considered successful if:

1. The app can run locally on a machine with an RTX 5080.
2. The system can analyze at least 20 curated coding problems with incorrect attempted solutions.
3. The model can provide useful bug diagnosis without immediately leaking full solutions.
4. Hint Mode avoids full-code leakage in most evaluation cases.
5. The evaluation pipeline can compare baseline and structured-prompt performance.
6. The codebase is modular enough to add QLoRA fine-tuning.
7. Benchmark results are reproducible.
8. The README explains the motivation, setup, model choice, dataset design, evaluation method, and limitations.
9. The project is understandable to recruiters and engineers within five minutes of reading the README.

## 18. Implementation Plan

### Phase 1: Scaffold

Create the repository structure, configuration files, and placeholder modules.

Deliverables:

1. Project folders.
2. `README.md`.
3. `PRD.md`.
4. `.env.example`.
5. `requirements.txt`.
6. `pyproject.toml`.
7. Basic test setup.

### Phase 2: Prompt and Inference MVP

Implement:

1. Prompt templates.
2. Model loader.
3. Local baseline inference.
4. Response parser.
5. FastAPI endpoint.
6. Streamlit UI.

### Phase 3: Dataset MVP

Implement:

1. JSONL dataset schema.
2. 100 initial examples.
3. Dataset cleaning script.
4. Train/validation split script.
5. Chat format converter.

### Phase 4: Evaluation MVP

Implement:

1. Leakage detector.
2. Response format checker.
3. Basic bug category matching.
4. Evaluation report generation.

### Phase 5: QLoRA Fine-Tuning

Implement:

1. QLoRA training script.
2. YAML training config.
3. Adapter saving.
4. Validation loss tracking.
5. Adapter loading for inference.

### Phase 6: Benchmarking

Implement:

1. Inference benchmark.
2. Memory benchmark.
3. CSV and Markdown result output.
4. README benchmark summary.

## 19. Implementation Notes for Code Agent

When implementing this project, follow these rules:

1. Create the project structure first.
2. Do not implement every feature at once.
3. Start with the MVP path.
4. Keep training, inference, evaluation, and app code separate.
5. Keep prompt templates separate from model-loading logic.
6. Use Pydantic schemas for API requests and responses.
7. Store model names and training settings in YAML config files.
8. Do not hardcode the model name across multiple files.
9. Keep functions small and testable.
10. Add unit tests for deterministic logic.
11. Use clear TODO comments only where future implementation is expected.
12. Avoid unsafe execution of user-submitted code in the MVP.
13. Write README instructions after the scaffold is created.
14. Make sure the project can run in baseline inference mode before adding fine-tuning.
15. If the 7B model causes memory issues, switch to the 3B fallback without changing the rest of the architecture.

## 20. Final Model Choice Summary

Use this model as the default:

```text
Qwen/Qwen2.5-Coder-7B-Instruct
```

Use this model as the fallback:

```text
Qwen/Qwen2.5-Coder-3B-Instruct
```

Do not start with larger MoE coding models or non-instruction code completion models. The goal is to build a reliable local fine-tuning and evaluation project, not to chase the largest possible model.
