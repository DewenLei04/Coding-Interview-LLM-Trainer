from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
import os


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"


@dataclass(frozen=True)
class GenerationConfig:
    model_name: str = DEFAULT_MODEL
    max_new_tokens: int = 700
    temperature: float = 0.2
    top_p: float = 0.9
    load_in_4bit: bool = True


class LLMBackend(ABC):
    backend_name: str
    model_name: str

    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class MockLLMBackend(LLMBackend):
    backend_name = "mock"

    def __init__(self, model_name: str = "mock-qwen-coder") -> None:
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        if "Mode: Bug Diagnosis" in prompt:
            return """## Diagnosis
The likely issue is in the window or state update logic. Use the failing test case to verify whether stale state is being reused.

## Likely Bug Location
Look near the condition that updates pointers, counters, or cached indexes.

## Why It Fails
The current state can become inconsistent with the active portion of the input, so later checks use outdated information.

## Failing Test Walkthrough
Step through each input element and write down every state change. The first divergence from the expected output identifies the bug.

## Suggested Fix
Guard the update so stale values outside the active range are ignored.

## Complexity
The intended optimized approach is usually O(n) time with O(k) auxiliary space for tracked state."""
        if "Mode: Progressive Hint" in prompt:
            return """## Hint
Focus on what information must be remembered between iterations, and what information becomes invalid as the search window moves.

## Why This Helps
Most interview bugs come from keeping state that was correct earlier but is no longer valid for the current candidate answer.

## Next Step
Trace the smallest failing example by hand and mark when each stored value should stop being trusted."""
        if "Mode: Complexity Explanation" in prompt:
            return """## Time Complexity
The time complexity depends on how many times each input element is visited. If each element is processed once, it is O(n).

## Space Complexity
The space complexity depends on the largest helper structure kept during execution.

## Variable Meaning
Identify which variables scale with input size and which remain constant.

## Cost Drivers
Loops, recursion depth, sorting, and growing maps or sets usually dominate cost.

## Corrections
If a nested loop does not always scan the full input, analyze the total number of pointer movements instead of multiplying loops blindly."""
        if "Mode: Full Solution" in prompt:
            return """## Algorithm Idea
Maintain the smallest state needed to decide each step, and update it monotonically as the input is scanned.

## Code
```python
def solve():
    raise NotImplementedError("Replace this scaffold with the problem-specific solution.")
```

## Explanation
The exact implementation depends on the problem constraints and examples.

## Complexity
After choosing the concrete algorithm, state the final Big-O costs, such as O(n) time and O(n) space for a single-pass stack solution.

## Edge Cases
Test empty input, one-element input, repeated values, and boundary-size cases."""
        return """## Interviewer Feedback
Good start. I would like you to justify the invariant your code maintains.

## Follow-up Questions
What edge case would break this approach? Can you explain why the core loop terminates?

## Edge Case Challenges
Try the smallest input, duplicated values, and values near the stated constraints.

## Complexity Prompt
Can you derive the runtime from how many times each element is visited?

## Testing Prompt
Create one passing case, one boundary case, and one case designed to challenge your main invariant."""


class TransformersLLMBackend(LLMBackend):
    backend_name = "transformers"

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self.model_name = config.model_name
        self._model = None
        self._tokenizer = None

    def _load_model_and_tokenizer(self):
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quantization_config = None
        if self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            quantization_config=quantization_config,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._model = model
        self._tokenizer = tokenizer
        return self._model, self._tokenizer

    def generate(self, prompt: str) -> str:
        import torch

        model, tokenizer = self._load_model_and_tokenizer()
        messages = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        if isinstance(encoded, Mapping):
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=model.device)
            else:
                attention_mask = attention_mask.to(model.device)
        else:
            input_ids = encoded.to(model.device)
            attention_mask = torch.ones_like(input_ids, device=model.device)
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.config.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if self.config.temperature > 0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                }
            )
        else:
            generation_kwargs["do_sample"] = False

        with torch.inference_mode():
            generated = model.generate(**generation_kwargs)
        new_tokens = generated[0, input_ids.shape[-1] :]
        return tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


def make_backend() -> LLMBackend:
    backend = os.getenv("CIT_BACKEND", "mock").strip().lower()
    model_name = os.getenv("CIT_MODEL_NAME", DEFAULT_MODEL)
    if backend == "transformers":
        return TransformersLLMBackend(
            GenerationConfig(
                model_name=model_name,
                max_new_tokens=int(os.getenv("CIT_MAX_NEW_TOKENS", "700")),
                temperature=float(os.getenv("CIT_TEMPERATURE", "0.2")),
                top_p=float(os.getenv("CIT_TOP_P", "0.9")),
                load_in_4bit=os.getenv("CIT_LOAD_IN_4BIT", "true").lower() == "true",
            )
        )
    return MockLLMBackend(model_name=os.getenv("CIT_MOCK_MODEL_NAME", "mock-qwen-coder"))
