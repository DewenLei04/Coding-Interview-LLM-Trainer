from pathlib import Path

from coding_interview_trainer.eval.validate_dataset import validate_dataset


def test_validate_seed_dataset_with_lower_minimum() -> None:
    report = validate_dataset(Path("data/seed_examples.jsonl"), minimum_rows=5)

    assert report["errors"] == 0
    assert report["valid_rows"] == 5
    assert report["ideal_response_rule_accuracy"] == 1.0


def test_validate_golden_eval_dataset() -> None:
    report = validate_dataset(Path("data/golden_eval.jsonl"), minimum_rows=2)

    assert report["errors"] == 0
    assert report["valid_rows"] == 2
