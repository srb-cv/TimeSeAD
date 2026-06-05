from collections import Counter
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from experiments_hydra.supervised.train_dsad import build_center_dataset_cfg
from timesead.data.sampler import BalancedBatchSampler
from timesead.data.transforms import Transform
from timesead.models.supervised import (
    DSADLoss,
    DSADSupervisionAnomalyDetector,
    DSADTargetTransform,
)


class _SingleSeriesSource(Transform):
    def __init__(self):
        super().__init__(None)
        self.inputs = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        self.targets = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0])

    def _get_datapoint_impl(self, item):
        return (self.inputs,), (self.targets,)

    def __len__(self):
        return 1

    @property
    def seq_len(self):
        return 8

    @property
    def num_features(self):
        return 2

    @property
    def ndim(self):
        return 2


def test_balanced_batch_sampler_yields_mixed_class_batches():
    labels = [0] * 8 + [1] * 3
    sampler = BalancedBatchSampler(labels=labels, batch_size=4)

    batches = list(sampler)

    assert len(batches) == 2
    for batch in batches:
        batch_labels = [labels[index] for index in batch]
        assert len(batch_labels) == 4
        assert Counter(batch_labels) == {0: 2, 1: 2}


def test_dsad_target_transform_preserves_input_window_labels():
    transform = DSADTargetTransform(_SingleSeriesSource(), window_size=3)

    inputs, targets = transform[0]

    assert inputs[0].shape == (3, 2)
    assert len(targets) == 1
    assert targets[0].shape == (3,)
    torch.testing.assert_close(inputs[0], torch.arange(6, dtype=torch.float32).reshape(3, 2))
    torch.testing.assert_close(targets[0], torch.tensor([0, 0, 0]))

    next_inputs, next_targets = transform[1]
    torch.testing.assert_close(next_inputs[0], torch.arange(6, 12, dtype=torch.float32).reshape(3, 2))
    torch.testing.assert_close(next_targets[0], torch.tensor([1, 1, 1]))


def test_dsad_loss_labels_window_anomalous_if_any_timestep_is_anomalous():
    criterion = DSADLoss(c=torch.zeros(1), reduction="none")
    reps = torch.tensor([[1.0], [2.0], [2.0]])
    labels = torch.tensor(
        [
            [0, 0, 0],  # normal: no anomalous timestep in the window
            [0, 1, 0],  # anomalous: one anomalous timestep marks the window anomalous
            [1, 1, 1],  # anomalous: all timesteps are anomalous
        ]
    )

    loss = criterion((reps,), (labels,))

    assert loss.shape == (3, 1)
    assert torch.isfinite(loss).all()
    torch.testing.assert_close(loss[1], loss[2])


def test_dsad_loss_accepts_scalar_window_labels():
    criterion = DSADLoss(c=torch.zeros(1), reduction="none")
    reps = torch.tensor([[1.0], [2.0]])
    labels = torch.tensor([0, 1])  # Already reduced to one label per window.

    loss = criterion((reps,), (labels,))

    assert loss.shape == (2, 1)
    assert torch.isfinite(loss).all()


def test_dsad_detector_preserves_chronological_score_and_label_order(monkeypatch):
    detector = DSADSupervisionAnomalyDetector(
        torch.nn.Identity(),
        DSADLoss(c=torch.zeros(1), reduction="none"),
        half_life=3,
    )

    def fake_compute_online_anomaly_score(inputs):
        b_inputs, _, moving_avg_num, moving_avg_denom = inputs
        scores = b_inputs[0][..., 0]
        return scores, moving_avg_num, moving_avg_denom

    monkeypatch.setattr(
        detector,
        "compute_online_anomaly_score",
        fake_compute_online_anomaly_score,
    )

    batch_1_inputs = torch.tensor(
        [
            [[10.0], [11.0], [12.0]],
            [[20.0], [21.0], [22.0]],
        ]
    )
    batch_1_labels = torch.tensor([[0, 1, 2], [3, 4, 5]])
    batch_2_inputs = torch.tensor([[[30.0], [31.0], [32.0]]])
    batch_2_labels = torch.tensor([[6, 7, 8]])
    batches = [
        ((batch_1_inputs,), (batch_1_labels,)),
        ((batch_2_inputs,), (batch_2_labels,)),
    ]

    labels, scores = detector.get_labels_and_scores(batches)

    torch.testing.assert_close(labels, torch.arange(9))
    torch.testing.assert_close(
        scores,
        torch.tensor([10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0]),
    )


def test_supervised_dmc_config_uses_new_dataset_flags_and_supervised_entrypoint():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "experiments_hydra"
        / "configs"
        / "dmc"
        / "supervised"
        / "train_dsad.yaml"
    )

    cfg = OmegaConf.load(config_path)

    assert "use_normal_only" not in cfg.dataset.ds_args
    assert cfg.dataset.ds_args.use_unsupervised_training is False
    assert cfg.dataset.ds_args.use_anomalous_as_normal is False
    assert cfg.training.supervised is True
    assert cfg.experiment.entrypoint == "experiments_hydra.supervised.train_dsad"
    assert cfg.experiment.evaluate_after_training is True
    assert cfg.experiment.post_train_evaluation.protocol == "holdout_calibration"
    assert cfg.experiment.post_train_evaluation.fit_detector_on == "saved_artifact"
    assert "lstm_hidden_dims" not in cfg.sweep.training_param_grid.model


def test_build_center_dataset_cfg_uses_normal_only_training_data():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "experiments_hydra"
        / "configs"
        / "dmc"
        / "supervised"
        / "train_dsad.yaml"
    )
    cfg = OmegaConf.load(config_path)

    center_cfg = build_center_dataset_cfg(cfg.dataset)

    assert center_cfg["name"] == cfg.dataset.name
    assert center_cfg["pipeline"] == OmegaConf.to_container(cfg.dataset.pipeline, resolve=True)
    assert center_cfg["split"] == OmegaConf.to_container(cfg.dataset.split, resolve=True)
    assert center_cfg["split_axis"] == cfg.dataset.split_axis
    assert center_cfg["ds_args"]["training"] is True
    assert center_cfg["ds_args"]["use_unsupervised_training"] is True
    assert center_cfg["ds_args"]["use_anomalous_as_normal"] is False
