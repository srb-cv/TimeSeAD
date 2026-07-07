from experiments_hydra.utils.sweep_adapter import (
    build_detector_hydra_overrides,
    build_hydra_overrides,
    build_training_hydra_overrides,
    combine_hydra_override_sets,
)


def test_build_training_hydra_overrides_excludes_detector_grid():
    spec = {
        "training_param_updates": {"training": {"epochs": 5}},
        "training_param_grid": {"model_params": {"dropout": [0.0, 0.1]}},
        "detector_param_grid": {"detector_params": {"half_life": [6, 10]}},
    }

    assert build_training_hydra_overrides(spec) == [
        {"training": {"epochs": 5}, "model": {"dropout": 0.0}},
        {"training": {"epochs": 5}, "model": {"dropout": 0.1}},
    ]


def test_build_detector_hydra_overrides_excludes_training_updates():
    spec = {
        "training_param_updates": {"training": {"epochs": 5}},
        "training_param_grid": {"model_params": {"dropout": [0.0, 0.1]}},
        "detector_param_grid": {"detector_params": {"half_life": [6, 10]}},
    }

    assert build_detector_hydra_overrides(spec) == [
        {"detector": {"half_life": 6}},
        {"detector": {"half_life": 10}},
    ]


def test_combine_hydra_override_sets_matches_legacy_combined_builder():
    spec = {
        "training_param_updates": {"training": {"epochs": 5}},
        "training_param_grid": {"model_params": {"dropout": [0.0, 0.1]}},
        "detector_param_grid": {"detector_params": {"half_life": [6, 10]}},
    }

    training_points = build_training_hydra_overrides(spec)
    detector_points = build_detector_hydra_overrides(spec)

    assert combine_hydra_override_sets(training_points, detector_points) == build_hydra_overrides(spec)


def test_build_hydra_overrides_defaults_to_single_empty_detector_point():
    spec = {
        "training_param_updates": {"training": {"epochs": 5}},
        "training_param_grid": {"model_params": {"dropout": [0.0, 0.1]}},
    }

    assert build_detector_hydra_overrides(spec) == [{}]
    assert build_hydra_overrides(spec) == build_training_hydra_overrides(spec)
