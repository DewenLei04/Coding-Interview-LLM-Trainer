from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coding_interview_trainer.agent.llm import FALLBACK_MODEL
from coding_interview_trainer.training.data import (
    SFTExample,
    load_sft_examples,
    render_chat_text,
    render_prompt_text,
)


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass(frozen=True)
class TrainConfig:
    dataset: Path
    output_dir: Path
    base_model: str
    max_seq_length: int
    epochs: float
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    min_examples: int
    allow_small_dataset: bool


class ChatSFTDataset:
    def __init__(
        self,
        examples: list[SFTExample],
        tokenizer: Any,
        max_seq_length: int,
    ) -> None:
        self.features = [
            self._tokenize_example(example, tokenizer, max_seq_length)
            for example in examples
        ]

    def _tokenize_example(
        self,
        example: SFTExample,
        tokenizer: Any,
        max_seq_length: int,
    ) -> dict[str, list[int]]:
        full_text = render_chat_text(tokenizer, example)
        prompt_text = render_prompt_text(tokenizer, example)
        full = tokenizer(
            full_text,
            max_length=max_seq_length,
            truncation=True,
            add_special_tokens=False,
        )
        prompt = tokenizer(
            prompt_text,
            max_length=max_seq_length,
            truncation=True,
            add_special_tokens=False,
        )
        input_ids = full["input_ids"]
        labels = input_ids.copy()
        prompt_length = min(len(prompt["input_ids"]), len(labels))
        labels[:prompt_length] = [-100] * prompt_length
        return {
            "input_ids": input_ids,
            "attention_mask": full["attention_mask"],
            "labels": labels,
        }

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.features[index]


class DataCollatorForChatSFT:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        pad_token_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}

        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_token_id] * pad_length)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_length)
            batch["labels"].append(feature["labels"] + [-100] * pad_length)

        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Qwen for interview tutoring.")
    parser.add_argument("--dataset", type=Path, default=Path("data/seed_examples.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/qwen25-coder-3b-interview-lora"))
    parser.add_argument("--base-model", default=FALLBACK_MODEL)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--min-examples", type=int, default=20)
    parser.add_argument("--allow-small-dataset", action="store_true")
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def train(config: TrainConfig) -> None:
    examples = load_sft_examples(config.dataset)
    if len(examples) < config.min_examples and not config.allow_small_dataset:
        raise SystemExit(
            f"Refusing to fine-tune on {len(examples)} examples. "
            f"Add more data or pass --allow-small-dataset for a smoke test."
        )

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = ChatSFTDataset(examples, tokenizer, config.max_seq_length)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        bf16=torch.cuda.is_available(),
        logging_steps=1,
        save_strategy="epoch",
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForChatSFT(tokenizer),
    )
    trainer.train()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
