from pathlib import Path

from coding_interview_trainer.eval.run_baseline import run_baseline


def test_run_baseline_writes_predictions_and_summary(tmp_path: Path) -> None:
    summary = run_baseline(
        Path("data/golden_eval.jsonl"),
        tmp_path,
        run_name="test_mock_baseline",
    )

    predictions_path = Path(summary["predictions_path"])
    summary_path = tmp_path / "test_mock_baseline_summary.json"

    assert summary["backend"] == "mock"
    assert summary["valid_rows"] == 10
    assert summary["rule_accuracy"] == 1.0
    assert predictions_path.exists()
    assert summary_path.exists()
