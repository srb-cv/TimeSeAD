import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from experiments_hydra.grid_search import (
    CandidateResult,
    SweepPlan,
    TrainedRunRef,
    run,
)


def _make_candidate(
    trained_run: TrainedRunRef,
    detector_idx: int,
    detector_overrides,
    selection_metric: str,
    selection_value: float,
) -> CandidateResult:
    combined = json.loads(json.dumps(trained_run.overrides))
    combined.update(detector_overrides)
    return CandidateResult(
        training_point=trained_run.training_point,
        detector_point=detector_idx,
        run_id=trained_run.run_id,
        run_dir=trained_run.run_dir,
        artifact_uri=trained_run.artifact_uri,
        training_overrides=trained_run.overrides,
        detector_overrides=detector_overrides,
        combined_overrides=combined,
        selection_metric=selection_metric,
        selection_value=selection_value,
        validation_metrics={
            "best_ts_f1_score": {"score": selection_value, "info": {}},
            "auprc": {"score": selection_value - 0.05, "info": {}},
        },
        validation_num_scores=10,
    )


def test_grid_search_smoke_matches_legacy_selection_flow(monkeypatch, tmp_path: Path):
    hydra_output_dir = tmp_path / "hydra_output"
    hydra_output_dir.mkdir()

    monkeypatch.setattr(
        "experiments_hydra.grid_search.HydraConfig.get",
        lambda: SimpleNamespace(
            runtime=SimpleNamespace(output_dir=str(hydra_output_dir))
        ),
    )
    monkeypatch.setattr(
        "experiments_hydra.grid_search.mlflow.set_tracking_uri",
        lambda uri: None,
    )

    class DummyClient:
        def __init__(self, tracking_uri=None):
            self.tracking_uri = tracking_uri

    monkeypatch.setattr("experiments_hydra.grid_search.MlflowClient", DummyClient)

    sweep_plan = SweepPlan(
        module_name="experiments_hydra.prediction.train_lstm_prediction_filonov",
        training_points=[
            {"model": {"dropout": 0.1}},
            {"model": {"dropout": 0.2}},
        ],
        detector_points=[
            {"detector": {"half_life": 4}},
            {"detector": {"half_life": 8}},
        ],
        selection_metric="best_ts_f1_score",
        evaluation_metrics=["best_ts_f1_score", "auprc"],
        mode="max",
        config_name_override=None,
        sweep_source="smoke-test",
    )
    monkeypatch.setattr(
        "experiments_hydra.grid_search._resolve_sweep_plan", lambda cfg: sweep_plan
    )

    training_calls = []

    def fake_train_candidates_once(cfg, plan, sweep_id, sweep_output_dir, client):
        training_calls.append(
            {
                "sweep_id": sweep_id,
                "num_training_points": len(plan.training_points),
                "num_detector_points": len(plan.detector_points),
                "sweep_output_dir": str(sweep_output_dir),
            }
        )
        return [
            TrainedRunRef(
                training_point=0,
                run_id="run-0",
                run_dir="/tmp/run-0",
                artifact_uri="file:///tmp/run-0/artifacts",
                overrides=plan.training_points[0],
            ),
            TrainedRunRef(
                training_point=1,
                run_id="run-1",
                run_dir="/tmp/run-1",
                artifact_uri="file:///tmp/run-1/artifacts",
                overrides=plan.training_points[1],
            ),
        ]

    monkeypatch.setattr(
        "experiments_hydra.grid_search._train_candidates_once",
        fake_train_candidates_once,
    )

    fold_call_state = {"index": 0}
    evaluation_calls = []
    fold_scores = {
        (0, 0): 0.2,
        (1, 1): 0.8,
        (0, 1): 0.4,
        (1, 0): 0.6,
    }
    fold_winners = [
        (0, 0),
        (1, 1),
        (0, 1),
        (1, 0),
        (0, 0),
    ]
    fold_validation_tables = [
        {(0, 0): 0.91, (0, 1): 0.52, (1, 0): 0.33, (1, 1): 0.28},
        {(0, 0): 0.24, (0, 1): 0.43, (1, 0): 0.60, (1, 1): 0.95},
        {(0, 0): 0.41, (0, 1): 0.89, (1, 0): 0.35, (1, 1): 0.61},
        {(0, 0): 0.37, (0, 1): 0.48, (1, 0): 0.92, (1, 1): 0.55},
        {(0, 0): 0.87, (0, 1): 0.62, (1, 0): 0.44, (1, 1): 0.30},
    ]
    final_selection_table = {
        (0, 0): 0.40,
        (0, 1): 0.52,
        (1, 0): 0.97,
        (1, 1): 0.68,
    }

    def fake_evaluate_candidates_for_fold(
        trained_runs,
        detector_points,
        splits,
        val_fold_index,
        selection_metric,
        metric_names,
        mode,
        fail_on_missing_metric,
        split_axis_override,
    ):
        evaluation_calls.append(
            {
                "splits": list(splits),
                "val_fold_index": val_fold_index,
                "num_trained_runs": len(trained_runs),
                "num_detector_points": len(detector_points),
            }
        )

        if list(splits) == [1]:
            score_table = final_selection_table
            expected_winner = (1, 0)
        else:
            fold_index = fold_call_state["index"]
            fold_call_state["index"] += 1
            score_table = fold_validation_tables[fold_index]
            expected_winner = fold_winners[fold_index]

        candidates = []
        best_candidate = None
        for trained_run in trained_runs:
            for detector_idx, detector_overrides in enumerate(detector_points):
                candidate = _make_candidate(
                    trained_run,
                    detector_idx,
                    detector_overrides,
                    selection_metric=selection_metric,
                    selection_value=score_table[
                        (trained_run.training_point, detector_idx)
                    ],
                )
                candidates.append(candidate)
                if (
                    trained_run.training_point,
                    detector_idx,
                ) == expected_winner:
                    best_candidate = candidate

        assert best_candidate is not None
        return best_candidate, candidates

    monkeypatch.setattr(
        "experiments_hydra.grid_search._evaluate_candidates_for_fold",
        fake_evaluate_candidates_for_fold,
    )

    def fake_score_candidate_on_test_folds(
        candidate,
        splits,
        test_fold_indices,
        metric_names,
        split_axis_override=None,
    ):
        base_score = fold_scores[(candidate.training_point, candidate.detector_point)]
        return {
            metric_name: [
                {
                    "score": base_score if metric_name == "best_ts_f1_score" else base_score + 0.05,
                    "info": {},
                    "test_fold": test_fold_index,
                    "num_scores": 10,
                }
                for test_fold_index in test_fold_indices
            ]
            for metric_name in metric_names
        }

    monkeypatch.setattr(
        "experiments_hydra.grid_search._score_candidate_on_test_folds",
        fake_score_candidate_on_test_folds,
    )

    cfg = OmegaConf.create(
        {
            "sweep": {
                "max_runs": None,
                "tracking_uri": None,
                "output_dir": None,
                "resume_trained_runs_from": None,
                "run_evaluation": True,
                "test_folds": 5,
                "padding": 1,
                "split_axis_override": None,
                "fail_on_missing_metric": True,
                "run_final_selection": True,
                "extra_overrides": [],
            }
        }
    )

    summary = run.__wrapped__(cfg)

    assert len(training_calls) == 1
    assert training_calls[0]["num_training_points"] == 2
    assert training_calls[0]["num_detector_points"] == 2

    assert len(evaluation_calls) == 6
    assert all(call["num_trained_runs"] == 2 for call in evaluation_calls)
    assert all(call["num_detector_points"] == 2 for call in evaluation_calls)

    assert len(summary["trained_runs"]) == 2
    assert len(summary["fold_results"]) == 5
    assert all(
        len(fold_result["all_validation_scores"]) == 4
        for fold_result in summary["fold_results"]
    )

    fold_winner_pairs = {
        (
            fold_result["best_candidate"]["training_point"],
            fold_result["best_candidate"]["detector_point"],
        )
        for fold_result in summary["fold_results"]
    }
    assert len(fold_winner_pairs) > 1

    assert summary["final_scores"]["best_ts_f1_score"] == pytest.approx(0.44)
    assert summary["final_scores"]["auprc"] == pytest.approx(0.49)

    assert summary["final_selection"]["best_candidate"]["training_point"] == 1
    assert summary["final_selection"]["best_candidate"]["detector_point"] == 0
    assert summary["best_run"] == summary["final_selection"]["best_candidate"]

    trained_runs_path = hydra_output_dir / "trained_runs.json"
    fold_results_path = hydra_output_dir / "fold_results.json"
    final_best_params_path = hydra_output_dir / "final_best_params.json"
    sweep_summary_path = hydra_output_dir / "sweep_summary.json"

    assert trained_runs_path.exists()
    assert fold_results_path.exists()
    assert final_best_params_path.exists()
    assert sweep_summary_path.exists()

    trained_runs_payload = json.loads(trained_runs_path.read_text(encoding="utf-8"))
    assert len(trained_runs_payload["trained_runs"]) == 2

    fold_results_payload = json.loads(fold_results_path.read_text(encoding="utf-8"))
    assert fold_results_payload["final_scores"]["best_ts_f1_score"] == pytest.approx(
        0.44
    )

    final_best_params_payload = json.loads(
        final_best_params_path.read_text(encoding="utf-8")
    )
    assert final_best_params_payload["final_best_params"] == {
        "model": {"dropout": 0.2},
        "detector": {"half_life": 4},
    }
