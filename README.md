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
- QLoRA fine-tuning pipeline and LoRA adapter loading.

The trained LoRA adapter is expected at `models/qwen25-coder-3b-interview-lora` when using the default fine-tuned inference commands.

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

## Run With Docker

Docker Compose can start both the FastAPI backend and Streamlit frontend with one command.

Quick mock demo, no GPU or model download:

```bash
docker compose up --build api frontend
```

Fine-tuned Qwen demo, using the saved LoRA adapter:

```bash
docker compose --profile gpu up --build api-gpu frontend-gpu
```

Then open:

```text
http://localhost:8501
```

Stop all Docker services:

```bash
docker compose down
```

The GPU command starts:

- FastAPI backend at `http://localhost:8000`
- Streamlit frontend at `http://localhost:8501`
- `Qwen/Qwen2.5-Coder-3B-Instruct`
- LoRA adapter loading from `models/qwen25-coder-3b-interview-lora`

Before using GPU mode, make sure the LoRA adapter exists locally:

```text
models/qwen25-coder-3b-interview-lora
```

GPU mode requires NVIDIA Container Toolkit and Docker GPU support. It mounts:

- `./models:/app/models:ro`
- a Docker volume for Hugging Face cache

Check the backend:

```bash
curl http://localhost:8000/health
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
Golden evaluation examples live in `data/golden_eval.jsonl` and should stay separate from examples used for training or fine-tuning.

Each example contains:

- `id`
- `request`
- `ideal_response`

The PRD target is at least 100 instruction examples. This MVP includes seed examples and the structure needed to grow the dataset.

## Evaluation and Benchmarking

Validate dataset JSONL, schema, duplicate IDs, mode coverage, and PRD-style response rules:

```bash
python3 -m coding_interview_trainer.eval.validate_dataset data/seed_examples.jsonl
```

Run the mock-mode evaluator:

```bash
python3 -m coding_interview_trainer.eval.evaluate_dataset data/seed_examples.jsonl
```

Run the benchmark:

```bash
python3 -m coding_interview_trainer.benchmarks.benchmark_inference --iterations 3
```

Run a persistent baseline inference pass:

```bash
python3 -m coding_interview_trainer.eval.run_baseline data/golden_eval.jsonl
```

This writes per-example predictions and a summary under `results/baselines/`.
Use the same command with `CIT_BACKEND=transformers` and `CIT_MODEL_NAME=...` for Qwen baseline runs.

## QLoRA Fine-Tuning

Install training dependencies:

```bash
python3 -m pip install -e ".[train]"
```

The trainer reads JSONL examples with `request` and `ideal_response`, renders the same tutor prompt used by the API, and trains only on the assistant response tokens.

First validate the training data:

```bash
python3 -m coding_interview_trainer.eval.validate_dataset data/seed_examples.jsonl
```

Run QLoRA fine-tuning on the 3B fallback model:

```bash
HF_HUB_DISABLE_XET=1 \
python3 -m coding_interview_trainer.training.train_qlora \
  --dataset data/seed_examples.jsonl \
  --base-model Qwen/Qwen2.5-Coder-3B-Instruct \
  --output-dir models/qwen25-coder-3b-interview-lora
```

The trainer refuses very small datasets by default. For a smoke test only:

```bash
HF_HUB_DISABLE_XET=1 \
python3 -m coding_interview_trainer.training.train_qlora \
  --dataset data/seed_examples.jsonl \
  --base-model Qwen/Qwen2.5-Coder-3B-Instruct \
  --output-dir models/qwen25-coder-3b-interview-lora-smoke \
  --allow-small-dataset
```

Run inference with a trained adapter:

```bash
HF_HUB_DISABLE_XET=1 \
CIT_BACKEND=transformers \
CIT_MODEL_NAME=Qwen/Qwen2.5-Coder-3B-Instruct \
CIT_ADAPTER_PATH=models/qwen25-coder-3b-interview-lora \
CIT_LOAD_IN_4BIT=true \
uvicorn coding_interview_trainer.api.main:app --reload
```

Compare fine-tuned output against the baseline:

```bash
HF_HUB_DISABLE_XET=1 \
CIT_BACKEND=transformers \
CIT_MODEL_NAME=Qwen/Qwen2.5-Coder-3B-Instruct \
CIT_ADAPTER_PATH=models/qwen25-coder-3b-interview-lora \
CIT_LOAD_IN_4BIT=true \
python3 -m coding_interview_trainer.eval.run_baseline \
  data/golden_eval.jsonl \
  --output-dir results/baselines \
  --run-name qwen25_coder_3b_lora_golden10_700tok
```

The validator checks:

- Valid JSONL rows.
- Required `request` and `ideal_response` fields.
- Pydantic request schema validity.
- Duplicate example IDs.
- Coverage across all five assistance modes.
- Required markdown sections per mode.
- Hint Mode and Bug Diagnosis code-leak rules.
- Full Solution code presence.
- Complexity notation where required.

## Tests

```bash
python3 -m pytest
```

## Current Limitations

- Safe code execution is not implemented yet.
- The frontend is a functional demo, not a polished production UI.
- Docker GPU mode depends on local NVIDIA Container Toolkit support.
