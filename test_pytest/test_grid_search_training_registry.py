import json
from pathlib import Path

from omegaconf import OmegaConf

from experiments_hydra.grid_search import (
    FoldResult,
    TrainedRunRef,
    _aggregate_final_scores,
    _extract_runtime_search_overrides,
    _load_training_registry,
    _merge_search_cfg,
    _resolve_sweep_plan,
    _resolve_selection_mode,
    _write_training_registry,
)


def test_resolve_selection_mode_defaults_to_max_for_f1_metrics():
    assert _resolve_selection_mode(None, None, "best_ts_f1_score") == "max"


def test_resolve_selection_mode_defaults_to_min_for_loss_metrics():
    assert _resolve_selection_mode(None, None, "val_loss_0") == "min"


def test_extract_runtime_search_overrides_ignores_null_entries():
    cfg = OmegaConf.create(
        {
            "training_param_updates": None,
            "training_param_grid": {"model": {"dropout": [0.1]}},
            "detector_param_grid": None,
            "selection_metric": "best_f1_score",
            "evaluation_metrics": None,
            "mode": None,
        }
    )

    assert _extract_runtime_search_overrides(cfg) == {
        "training_param_grid": {"model": {"dropout": [0.1]}},
        "selection_metric": "best_f1_score",
    }


def test_merge_search_cfg_recursively_merges_runtime_overrides():
    merged = _merge_search_cfg(
        {
            "training_param_updates": {
                "dataset": {"name": "MiniSMDDataset"},
                "training": {"epochs": 100},
            },
            "training_param_grid": {
                "model": {"hidden_dims": [[16]], "latent_dim": [20, 50]}
            },
        },
        {
            "training_param_updates": {"training": {"epochs": 1}},
            "training_param_grid": {"model": {"latent_dim": [4, 8]}},
        },
    )

    assert merged == {
        "training_param_updates": {
            "dataset": {"name": "MiniSMDDataset"},
            "training": {"epochs": 1},
        },
        "training_param_grid": {
            "model": {"latent_dim": [4, 8]}
        },
    }


def test_merge_search_cfg_replaces_runtime_grids_instead_of_deep_merging():
    merged = _merge_search_cfg(
        {
            "training_param_grid": {
                "model": {"hidden_dims": [[16]], "latent_dim": [20, 50]},
                "training": {"optimizer": {"args": {"lr": [1.0e-4, 1.0e-3]}}},
            },
            "detector_param_grid": {
                "detector": {"num_mc_samples": [32, 64]}
            },
        },
        {
            "training_param_grid": {
                "model": {"latent_dim": [4, 8]},
            },
            "detector_param_grid": {
                "detector": {"num_mc_samples": [2, 4]}
            },
        },
    )

    assert merged == {
        "training_param_grid": {
            "model": {"latent_dim": [4, 8]},
        },
        "detector_param_grid": {
            "detector": {"num_mc_samples": [2, 4]}
        },
    }


def test_resolve_sweep_plan_merges_runtime_overrides_with_hydra_experiment_config(
    monkeypatch,
):
    experiment_cfg = OmegaConf.create(
        {
            "experiment": {
                "entrypoint": "experiments_hydra.generative.vae.train_donut"
            },
            "sweep": {
                "validation_metric": "best_ts_f1_score",
                "evaluation_metrics": ["best_ts_f1_score", "auprc"],
                "mode": "max",
                "training_param_updates": {
                    "dataset": {"name": "MiniSMDDataset"},
                    "training": {"epochs": 100},
                },
                "training_param_grid": {
                    "model": {
                        "hidden_dims": [[16]],
                        "latent_dim": [20, 50],
                        "mask_prob": [0.03],
                    },
                    "training": {
                        "optimizer": {"args": {"lr": [1.0e-4, 1.0e-3]}}
                    },
                },
                "detector_param_grid": {
                    "detector": {"num_mc_samples": [32, 64]}
                },
            },
        }
    )
    monkeypatch.setattr(
        "experiments_hydra.grid_search._load_hydra_experiment_cfg",
        lambda config_name: experiment_cfg,
    )

    cfg = OmegaConf.create(
        {
            "sweep": {
                "experiment_config": "parity/mini_smd_donut",
                "config_path": None,
                "training_param_updates": {"training": {"epochs": 1}},
                "training_param_grid": {
                    "model": {"latent_dim": [4, 8], "mask_prob": [0.01]},
                    "dataset": {
                        "pipeline": {"window": {"args": {"window_size": [10]}}}
                    },
                },
                "detector_param_grid": {
                    "detector": {"num_mc_samples": [2, 4]}
                },
                "selection_metric": "best_f1_score",
                "evaluation_metrics": ["best_f1_score"],
                "mode": None,
            }
        }
    )

    plan = _resolve_sweep_plan(cfg)

    assert plan.module_name == "experiments_hydra.generative.vae.train_donut"
    assert plan.selection_metric == "best_f1_score"
    assert plan.evaluation_metrics == ["best_f1_score"]
    assert plan.mode == "max"
    assert plan.training_points == [
        {
            "dataset": {
                "name": "MiniSMDDataset",
                "pipeline": {"window": {"args": {"window_size": 10}}},
            },
            "training": {"epochs": 1},
            "model": {
                "latent_dim": 4,
                "mask_prob": 0.01,
            },
        },
        {
            "dataset": {
                "name": "MiniSMDDataset",
                "pipeline": {"window": {"args": {"window_size": 10}}},
            },
            "training": {"epochs": 1},
            "model": {
                "latent_dim": 8,
                "mask_prob": 0.01,
            },
        },
    ]
    assert plan.detector_points == [
        {"detector": {"num_mc_samples": 2}},
        {"detector": {"num_mc_samples": 4}},
    ]


def test_resolve_sweep_plan_merges_runtime_overrides_with_legacy_spec(
    monkeypatch,
):
    monkeypatch.setattr(
        "experiments_hydra.grid_search.load_sweep_spec",
        lambda path: {
            "params": {
                "training_experiment": "prediction.train_lstm_prediction_filonov",
                "validation_metric": "best_ts_f1_score",
                "evaluation_metrics": ["best_ts_f1_score", "auprc"],
                "mode": "max",
            },
            "training_param_updates": {
                "dataset": {"name": "SMDDataset"},
                "training": {"epochs": 100},
            },
            "training_param_grid": {
                "model_params": {"dropout": [0.0, 0.2]},
            },
            "detector_param_grid": {
                "detector_params": {"half_life": [6, 10]},
            },
        },
    )

    cfg = OmegaConf.create(
        {
            "sweep": {
                "experiment_config": None,
                "config_path": "experiment_configs/smd/prediction/train_lstm_prediction_filonov_on_smd.yml",
                "training_param_updates": {"training": {"epochs": 1}},
                "training_param_grid": {"model": {"dropout": [0.1]}},
                "detector_param_grid": {"detector": {"half_life": [4]}},
                "selection_metric": "best_f1_score",
                "evaluation_metrics": ["best_f1_score"],
                "mode": None,
            }
        }
    )

    plan = _resolve_sweep_plan(cfg)

    assert (
        plan.module_name
        == "experiments_hydra.prediction.train_lstm_prediction_filonov"
    )
    assert plan.selection_metric == "best_f1_score"
    assert plan.evaluation_metrics == ["best_f1_score"]
    assert plan.mode == "max"
    assert plan.training_points == [
        {
            "dataset": {"name": "SMDDataset"},
            "training": {"epochs": 1},
            "model": {"dropout": 0.1},
        }
    ]
    assert plan.detector_points == [{"detector": {"half_life": 4}}]


def test_training_registry_roundtrip(tmp_path: Path):
    registry_path = tmp_path / "trained_runs.json"
    trained_runs = [
        TrainedRunRef(
            training_point=0,
            run_id="run-0",
            run_dir="/tmp/train_000",
            artifact_uri="file:///tmp/artifacts/0",
            overrides={"model": {"dropout": 0.1}},
        ),
        TrainedRunRef(
            training_point=1,
            run_id="run-1",
            run_dir="/tmp/train_001",
            artifact_uri="file:///tmp/artifacts/1",
            overrides={"model": {"dropout": 0.2}},
        ),
    ]

    _write_training_registry(registry_path, trained_runs)
    loaded_runs = _load_training_registry(registry_path)

    assert loaded_runs == trained_runs


def test_training_registry_file_shape(tmp_path: Path):
    registry_path = tmp_path / "trained_runs.json"
    trained_runs = [
        TrainedRunRef(
            training_point=0,
            run_id="run-0",
            run_dir="/tmp/train_000",
            artifact_uri="file:///tmp/artifacts/0",
            overrides={"model": {"dropout": 0.1}},
        )
    ]

    _write_training_registry(registry_path, trained_runs)

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    assert payload == {
        "trained_runs": [
            {
                "training_point": 0,
                "run_id": "run-0",
                "run_dir": "/tmp/train_000",
                "artifact_uri": "file:///tmp/artifacts/0",
                "overrides": {"model": {"dropout": 0.1}},
            }
        ]
    }


def test_aggregate_final_scores_averages_fold_test_scores():
    fold_results = [
        FoldResult(
            val_fold=0,
            splits=[1, 1, 3],
            val_fold_index=1,
            test_fold_indices=[2],
            best_candidate=None,
            test_scores={
                "best_ts_f1_score": [
                    {"score": 0.6, "info": {}, "test_fold": 2, "num_scores": 10}
                ],
                "auprc": [
                    {"score": 0.5, "info": {}, "test_fold": 2, "num_scores": 10}
                ],
            },
            all_validation_scores=[],
        ),
        FoldResult(
            val_fold=1,
            splits=[1, 1, 1, 2],
            val_fold_index=1,
            test_fold_indices=[3],
            best_candidate=None,
            test_scores={
                "best_ts_f1_score": [
                    {"score": 0.4, "info": {}, "test_fold": 3, "num_scores": 20}
                ],
                "auprc": [
                    {"score": 0.7, "info": {}, "test_fold": 3, "num_scores": 20}
                ],
            },
            all_validation_scores=[],
        ),
    ]

    final_scores = _aggregate_final_scores(
        fold_results, ["best_ts_f1_score", "auprc"], total_folds=2
    )

    assert final_scores == {
        "best_ts_f1_score": 0.5,
        "auprc": 0.6,
    }
