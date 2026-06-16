from pathlib import Path

from coding_interview_trainer.training.data import load_sft_examples


def test_load_sft_examples_builds_prompt_and_response() -> None:
    examples = load_sft_examples(Path("data/golden_eval.jsonl"))

    assert len(examples) == 10
    first = examples[0]
    assert first.example_id.startswith("golden-")
    assert "Mode:" in first.prompt
    assert "Interview context:" in first.prompt
    assert first.response.startswith("## ")
    assert first.messages == [
        {"role": "user", "content": first.prompt},
        {"role": "assistant", "content": first.response},
    ]
