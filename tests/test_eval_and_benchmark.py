from pathlib import Path

from coding_interview_trainer.benchmarks.benchmark_inference import benchmark
from coding_interview_trainer.eval.evaluate_dataset import evaluate


def test_eval_seed_dataset() -> None:
    metrics = evaluate(Path("data/seed_examples.jsonl"))

    assert metrics["valid_rows"] >= 5
    assert metrics["generated_rule_accuracy"] == 1.0
    assert metrics["ideal_response_rule_accuracy"] == 1.0


def test_benchmark_mock_mode() -> None:
    metrics = benchmark(iterations=1)

    assert metrics["backend"] == "mock"
    assert metrics["iterations"] == 1
