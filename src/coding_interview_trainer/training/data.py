from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from coding_interview_trainer.prompts.templates import build_prompt
from coding_interview_trainer.schemas import InterviewRequest


@dataclass(frozen=True)
class SFTExample:
    example_id: str
    prompt: str
    response: str

    @property
    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "user", "content": self.prompt},
            {"role": "assistant", "content": self.response},
        ]


def load_sft_examples(path: Path) -> list[SFTExample]:
    examples: list[SFTExample] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = row.get("id") or f"line-{line_number}"
            request_payload: dict[str, Any] | None = row.get("request")
            response = row.get("ideal_response")
            if not isinstance(request_payload, dict):
                raise ValueError(f"line {line_number}: missing request object")
            if not isinstance(response, str) or not response.strip():
                raise ValueError(f"line {line_number}: missing ideal_response text")
            try:
                request = InterviewRequest.model_validate(request_payload)
            except ValidationError as exc:
                raise ValueError(f"line {line_number}: invalid request: {exc}") from exc
            examples.append(
                SFTExample(
                    example_id=str(example_id),
                    prompt=build_prompt(request),
                    response=response.strip(),
                )
            )
    return examples


def render_chat_text(tokenizer: Any, example: SFTExample) -> str:
    return tokenizer.apply_chat_template(
        example.messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def render_prompt_text(tokenizer: Any, example: SFTExample) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": example.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
