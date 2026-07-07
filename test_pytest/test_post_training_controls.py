from pathlib import Path

from omegaconf import OmegaConf

from experiments_hydra.utils.post_training import (
    maybe_evaluate_after_training,
    should_evaluate_after_training,
)


def test_should_evaluate_after_training_defaults_false():
    cfg = OmegaConf.create({"experiment": {}})

    assert should_evaluate_after_training(cfg) is False


def test_maybe_evaluate_after_training_skips_when_disabled(monkeypatch, tmp_path: Path):
    called = False

    def fake_evaluate_run(*args, **kwargs):
        nonlocal called
        called = True
        return {"status": "called"}

    monkeypatch.setattr(
        "experiments_hydra.utils.post_training.evaluate_run",
        fake_evaluate_run,
    )
    cfg = OmegaConf.create({"experiment": {"evaluate_after_training": False}})

    assert maybe_evaluate_after_training(cfg, tmp_path) is None
    assert called is False


def test_maybe_evaluate_after_training_calls_evaluate(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_evaluate_run(*, run_dir):
        captured["run_dir"] = run_dir
        return {"status": "called"}

    monkeypatch.setattr(
        "experiments_hydra.utils.post_training.evaluate_run",
        fake_evaluate_run,
    )
    cfg = OmegaConf.create({"experiment": {"evaluate_after_training": True}})

    result = maybe_evaluate_after_training(cfg, tmp_path)

    assert result == {"status": "called"}
    assert captured["run_dir"] == tmp_path


def test_maybe_evaluate_after_training_calls_holdout_protocol(
    monkeypatch, tmp_path: Path
):
    captured = {}

    def fake_evaluate_run_with_holdout(*, run_dir, threshold_metric, fit_detector_on):
        captured["run_dir"] = run_dir
        captured["threshold_metric"] = threshold_metric
        captured["fit_detector_on"] = fit_detector_on
        return {"status": "holdout"}

    monkeypatch.setattr(
        "experiments_hydra.utils.post_training.evaluate_run_with_holdout",
        fake_evaluate_run_with_holdout,
    )
    cfg = OmegaConf.create(
        {
            "experiment": {
                "evaluate_after_training": True,
                "post_train_evaluation": {
                    "protocol": "holdout_calibration",
                    "threshold_metric": "best_ts_f1_score",
                    "fit_detector_on": "training_validation",
                },
            }
        }
    )

    result = maybe_evaluate_after_training(cfg, tmp_path)

    assert result == {"status": "holdout"}
    assert captured == {
        "run_dir": tmp_path,
        "threshold_metric": "best_ts_f1_score",
        "fit_detector_on": "training_validation",
    }
