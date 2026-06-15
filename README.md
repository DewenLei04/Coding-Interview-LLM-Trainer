# Coding-Interview-LLM-Trainer

Build a local LLM-powered coding interview coach that can analyze a user's coding interview solution, identify bugs, provide progressive hints, explain time and space complexity, and only provide the full solution when explicitly requested.

## MVP Status

This repo contains the first MVP scaffold:

- FastAPI backend with `POST /analyze`.
- Streamlit frontend for interactive practice.
- Typed request/response schemas for five assistance modes.
- Prompt templates that enforce the PRD tutoring rules.
- Mock inference backend for development without model downloads.
- Optional Transformers backend for local Qwen inference.
- JSONL dataset schema and seed examples.
- Evaluation and benchmark scripts.
- Unit/API tests for the core behavior.

Fine-tuning is intentionally not implemented in this pass. The package structure leaves room for QLoRA training modules later.

## Assistance Modes

- `bug_diagnosis`
- `progressive_hint`
- `complexity_explanation`
- `full_solution`
- `interview_simulation`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
```

For local 4-bit Qwen inference, install the GPU extra too:

```bash
python3 -m pip install -e ".[dev,gpu]"
```

For non-quantized local Transformers inference, install:

```bash
python3 -m pip install -e ".[dev,llm]"
```

## Run the API

Mock mode is the default and does not download a model:

```bash
uvicorn coding_interview_trainer.api.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Analyze example:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "problem_title": "Longest Substring Without Repeating Characters",
    "problem_description": "Given a string s, return the length of the longest substring without repeating characters.",
    "programming_language": "Python",
    "mode": "bug_diagnosis",
    "attempted_solution": "def f(s): return 0",
    "failing_test_case": "s = \"abba\"",
    "expected_output": "2",
    "wrong_output": "3"
  }'
```

## Run the Frontend

Start the API first, then run:

```bash
streamlit run frontend/app.py
```

Set a custom API URL if needed:

```bash
CIT_API_URL=http://localhost:8000 streamlit run frontend/app.py
```

## Use Local Qwen Inference

The default model is `Qwen/Qwen2.5-Coder-7B-Instruct`. The PRD fallback model is `Qwen/Qwen2.5-Coder-3B-Instruct`.

```bash
CIT_BACKEND=transformers \
CIT_MODEL_NAME=Qwen/Qwen2.5-Coder-7B-Instruct \
CIT_LOAD_IN_4BIT=true \
uvicorn coding_interview_trainer.api.main:app --reload
```

Fallback example:

```bash
CIT_BACKEND=transformers \
CIT_MODEL_NAME=Qwen/Qwen2.5-Coder-3B-Instruct \
uvicorn coding_interview_trainer.api.main:app --reload
```

Useful environment variables:

- `CIT_BACKEND`: `mock` or `transformers`.
- `CIT_MODEL_NAME`: Hugging Face model ID.
- `CIT_LOAD_IN_4BIT`: `true` or `false`.
- `CIT_MAX_NEW_TOKENS`: generation limit.
- `CIT_TEMPERATURE`: generation temperature.
- `CIT_TOP_P`: nucleus sampling value.

## Dataset

Dataset examples live in `data/seed_examples.jsonl`. The schema is documented in `data/schema.md`.

Each example contains:

- `id`
- `request`
- `ideal_response`

The PRD target is at least 100 instruction examples. This MVP includes seed examples and the structure needed to grow the dataset.

## Evaluation and Benchmarking

Run the mock-mode evaluator:

```bash
python3 -m coding_interview_trainer.eval.evaluate_dataset data/seed_examples.jsonl
```

Run the benchmark:

```bash
python3 -m coding_interview_trainer.benchmarks.benchmark_inference --iterations 3
```

## Tests

```bash
python3 -m pytest
```

## Current Limitations

- QLoRA fine-tuning is not implemented yet.
- Safe code execution is not implemented yet.
- The frontend is a functional demo, not a polished production UI.
- The seed dataset is intentionally small and should be expanded before fine-tuning.
