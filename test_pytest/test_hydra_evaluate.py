from types import SimpleNamespace

import pytest
import torch

from experiments_hydra.evaluate import (
    _maybe_log_to_mlflow,
    build_detector_from_artifact,
    build_labelled_test_dataset_cfg,
    compute_val_splits,
    evaluate_metric_set,
    evaluate_run_with_holdout,
    fit_detector_if_supported,
    select_threshold,
)
from timesead.models.baselines import IQRAnomalyDetector
from timesead.models.prediction import LSTMPredictionAnomalyDetector
from timesead.models.prediction.lstm_prediction import (
    LSTMS2SPredictionAnomalyDetector,
    halflife2alpha,
)


@pytest.mark.parametrize(
    ("folds", "val_fold", "padding", "expected"),
    [
        (1, 0, 1, ([1], 0, [])),
        (5, 0, 1, ([1, 1, 3], 0, [2])),
        (5, 1, 1, ([1, 1, 1, 2], 1, [3])),
        (5, 2, 1, ([1, 1, 1, 1, 1], 2, [0, 4])),
        (5, 3, 1, ([2, 1, 1, 1], 2, [0])),
        (5, 4, 1, ([3, 1, 1], 2, [0])),
    ],
)
def test_compute_val_splits_matches_legacy_behavior(
    folds, val_fold, padding, expected
):
    assert compute_val_splits(folds, val_fold, padding) == expected


def test_compute_val_splits_rejects_invalid_fold_count():
    with pytest.raises(ValueError):
        compute_val_splits(0, 0)


def test_compute_val_splits_rejects_invalid_validation_index():
    with pytest.raises(ValueError):
        compute_val_splits(3, 3)


def test_build_detector_from_artifact_reconstructs_prediction_detector():
    training_cfg = {
        "experiment": {"family": "prediction"},
        "training": {"device": "cpu"},
        "dataset": {"pipeline": {"prediction": {"args": {"window_size": 5}}}},
    }
    payload = {"model": torch.nn.Identity(), "detector": None}

    detector = build_detector_from_artifact(
        payload,
        training_cfg,
        detector_overrides={"half_life": 11},
        prefer_saved_detector=False,
    )

    assert isinstance(detector, LSTMS2SPredictionAnomalyDetector)
    assert detector.alpha == pytest.approx(halflife2alpha(11))


def test_build_detector_from_artifact_reconstructs_baseline_detector():
    training_cfg = {
        "experiment": {"family": "baseline"},
        "training": {"device": "cpu"},
        "dataset": {"pipeline": {}},
        "detector": {
            "std_factor": 2.58,
            "first_diffs": True,
            "cum_method": "mean",
            "feature_index": None,
            "input_shape": "btf",
        },
    }
    payload = {"model": None, "detector": None}

    detector = build_detector_from_artifact(
        payload,
        training_cfg,
        detector_overrides={"detector": {"std_factor": 1.28, "input_shape": "tbf"}},
        prefer_saved_detector=False,
    )

    assert isinstance(detector, IQRAnomalyDetector)
    assert detector.std_factor == pytest.approx(1.28)
    assert detector.input_shape == "tbf"
    assert detector.first_diffs is True


def test_build_detector_from_artifact_prefers_saved_detector():
    saved_detector = torch.nn.Linear(1, 1)
    training_cfg = {
        "experiment": {"family": "prediction"},
        "training": {"device": "cpu"},
        "dataset": {"pipeline": {"prediction": {"args": {"window_size": 5}}}},
    }
    payload = {"model": torch.nn.Identity(), "detector": saved_detector}

    detector = build_detector_from_artifact(payload, training_cfg)

    assert detector is saved_detector


def test_build_labelled_test_dataset_cfg_applies_pipeline_overrides_to_test_pipeline():
    training_cfg = {
        "dataset": {
            "split": [0.75, 0.25],
            "split_axis": "time",
            "ds_args": {"training": True},
            "pipeline": {
                "prediction": {"args": {"window_size": 32, "step_size": 1}},
                "normalize": {"args": {"method": "meanstd"}},
            },
            "test_pipeline": {
                "prediction": {"args": {"window_size": 8, "step_size": 1}},
                "normalize": {"args": {"method": "meanstd"}},
            },
        }
    }

    dataset_cfg = build_labelled_test_dataset_cfg(
        training_cfg,
        split=[1, 1, 3],
        pipeline_overrides={"prediction": {"args": {"window_size": 32}}},
    )

    assert dataset_cfg["ds_args"]["training"] is False
    assert dataset_cfg["pipeline"]["prediction"]["args"]["window_size"] == 32
    assert dataset_cfg["pipeline"]["normalize"]["args"]["method"] == "meanstd"
    assert dataset_cfg["split"] == [1, 1, 3]
    assert dataset_cfg["split_axis"] == "time"


def test_fit_detector_if_supported_passes_through_kwargs():
    class DummyDetector:
        def __init__(self):
            self.calls = []

        def fit(self, loader, **kwargs):
            self.calls.append((loader, kwargs))

    detector = DummyDetector()
    loader = object()

    fit_detector_if_supported(detector, loader, fit_kwargs={"foo": 1})

    assert detector.calls == [(loader, {"foo": 1})]


def test_fit_detector_if_supported_supplies_prediction_fit_kwargs():
    class DummyPredictionDetector(LSTMPredictionAnomalyDetector):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.register_buffer("dummy", torch.tensor([]), persistent=False)
            self.calls = []

        def fit(self, dataset, *, subseq_lengths, window_size, **kwargs):
            self.calls.append(
                (
                    dataset,
                    {
                        "subseq_lengths": subseq_lengths,
                        "window_size": window_size,
                        **kwargs,
                    },
                )
            )

        def compute_online_anomaly_score(self, inputs):
            raise NotImplementedError

        def compute_offline_anomaly_score(self, inputs):
            raise NotImplementedError

        def format_online_targets(self, targets):
            raise NotImplementedError

    loader = SimpleNamespace(
        dataset=SimpleNamespace(
            sink_transform=SimpleNamespace(input_window_size=8, parent=None),
            seq_len=[10, 12],
        )
    )
    detector = DummyPredictionDetector()

    fit_detector_if_supported(detector, loader)

    assert detector.calls == [
        (loader, {"subseq_lengths": [10, 12], "window_size": 8})
    ]


def test_evaluate_metric_set_applies_fixed_threshold_without_reoptimizing():
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    scores = torch.tensor([0.3, 0.85, 0.7, 0.75], dtype=torch.float)
    training_cfg = {
        "sweep": {
            "validation_metric": "best_f1_score",
            "evaluation_metrics": ["best_f1_score", "auprc"],
        }
    }

    metrics = evaluate_metric_set(
        labels,
        scores,
        training_cfg,
        applied_threshold=0.8,
    )

    assert set(metrics) == {"f1_score", "auprc"}
    assert metrics["f1_score"]["score"] == pytest.approx(2 / 3)
    assert metrics["f1_score"]["info"]["threshold"] == pytest.approx(0.8)


def test_evaluate_run_with_holdout_tunes_on_one_split_and_reports_on_other(
    monkeypatch, tmp_path
):
    training_cfg = {
        "training": {"device": "cpu"},
        "dataset": {"split": [0.3, 0.7], "split_axis": "time"},
        "sweep": {
            "validation_metric": "best_f1_score",
            "evaluation_metrics": ["best_f1_score", "auprc"],
        },
    }
    tune_labels = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    tune_scores = torch.tensor([0.1, 0.9, 0.8, 0.2], dtype=torch.float)
    eval_labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    eval_scores = torch.tensor([0.3, 0.85, 0.7, 0.75], dtype=torch.float)

    monkeypatch.setattr(
        "experiments_hydra.evaluate.load_training_cfg",
        lambda run_dir: training_cfg,
    )
    monkeypatch.setattr(
        "experiments_hydra.evaluate.build_labelled_test_splits",
        lambda *args, **kwargs: (
            {"split": [0.3, 0.7], "split_axis": "time"},
            ["tune_split", "eval_split"],
        ),
    )
    monkeypatch.setattr(
        "experiments_hydra.evaluate._build_detector_for_holdout",
        lambda *args, **kwargs: (object(), {"source": "training_validation"}),
    )

    def fake_collect_labels_and_scores(detector, dataset, training_cfg):
        if dataset == "tune_split":
            return tune_labels, tune_scores
        if dataset == "eval_split":
            return eval_labels, eval_scores
        raise AssertionError(f"Unexpected dataset {dataset!r}")

    monkeypatch.setattr(
        "experiments_hydra.evaluate.collect_labels_and_scores",
        fake_collect_labels_and_scores,
    )

    summary = evaluate_run_with_holdout(
        run_dir=tmp_path,
        threshold_metric="best_f1_score",
        log_to_mlflow=False,
    )

    assert summary["protocol"] == "holdout_calibration"
    assert summary["tuning"]["split_index"] == 0
    assert summary["evaluation"]["split_index"] == 1
    assert summary["selected_threshold"] == pytest.approx(
        select_threshold(tune_labels, tune_scores, "best_f1_score")["threshold"]
    )
    assert summary["metrics"]["f1_score"]["score"] == pytest.approx(2 / 3)
    assert (tmp_path / "evaluation_summary.json").exists()


def test_maybe_log_to_mlflow_reuses_active_matching_run(monkeypatch, tmp_path):
    class ActiveRunInfo:
        run_id = "run-123"

    class ActiveRun:
        info = ActiveRunInfo()

    calls = {"tags": [], "metrics": []}
    monkeypatch.setattr(
        "experiments_hydra.evaluate.mlflow.set_tracking_uri",
        lambda uri: calls.__setitem__("tracking_uri", uri),
    )
    monkeypatch.setattr(
        "experiments_hydra.evaluate.mlflow.active_run",
        lambda: ActiveRun(),
    )

    def fail_start_run(*args, **kwargs):
        raise AssertionError("start_run must not be called for active matching run")

    monkeypatch.setattr("experiments_hydra.evaluate.mlflow.start_run", fail_start_run)
    monkeypatch.setattr(
        "experiments_hydra.evaluate.log_hydra_run_reference",
        lambda run_dir: calls.__setitem__("hydra_run_dir", str(run_dir)),
    )
    monkeypatch.setattr(
        "experiments_hydra.evaluate.mlflow.set_tag",
        lambda key, value: calls["tags"].append((key, value)),
    )
    monkeypatch.setattr(
        "experiments_hydra.evaluate.mlflow.log_metric",
        lambda name, value: calls["metrics"].append((name, value)),
    )

    summary_path = tmp_path / "evaluation_summary.json"
    summary = {
        "metrics": {
            "f1_score": {"score": 0.5, "info": {}},
            "auprc": {"score": None, "info": {}},
        }
    }
    _maybe_log_to_mlflow(
        training_cfg={"experiment": {"tracking_uri": None}},
        cli_cfg={
            "run_dir": str(tmp_path),
            "run_id": "run-123",
            "tracking_uri": "sqlite:///tmp/mlflow.db",
            "log_to_mlflow": True,
        },
        summary_path=summary_path,
        summary=summary,
    )

    assert calls["tracking_uri"] == "sqlite:///tmp/mlflow.db"
    assert calls["hydra_run_dir"] == str(tmp_path)
    assert calls["metrics"] == [("f1_score", 0.5)]
    assert calls["tags"] == [("hydra_evaluation_summary", str(summary_path.resolve()))]
