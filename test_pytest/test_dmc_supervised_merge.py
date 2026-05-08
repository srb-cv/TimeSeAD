from collections import Counter
from pathlib import Path

import torch
from omegaconf import OmegaConf

from timesead.data.sampler import BalancedBatchSampler
from timesead.data.transforms import Transform
from timesead.models.supervised import DSADLoss, DSADTargetTransform


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


def test_dsad_target_transform_preserves_labels_and_adds_future_target_window():
    transform = DSADTargetTransform(_SingleSeriesSource(), window_size=3)

    inputs, targets = transform[0]

    assert inputs[0].shape == (3, 2)
    assert targets[0].shape == (3,)
    assert targets[1].shape == (3, 2)
    torch.testing.assert_close(inputs[0], torch.arange(6, dtype=torch.float32).reshape(3, 2))
    torch.testing.assert_close(targets[0], torch.tensor([1, 1, 1]))
    torch.testing.assert_close(targets[1], torch.arange(6, 12, dtype=torch.float32).reshape(3, 2))


def test_dsad_loss_accepts_window_labels_and_representations():
    criterion = DSADLoss(c=torch.zeros(2), reduction="none")
    reps = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    labels = torch.tensor([[0, 0, 0], [1, 1, 1]])

    loss = criterion((reps,), (labels,))

    assert loss.shape == (2, 1)
    assert torch.isfinite(loss).all()


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
    assert "lstm_hidden_dims" not in cfg.sweep.training_param_grid.model
