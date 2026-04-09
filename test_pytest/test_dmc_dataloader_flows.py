import os
from pathlib import Path

import pytest
import torch
import numpy as np

from timesead.data.dmc_dataset import DMCDataset   # adjust import if needed
from timesead.data.preprocessing.dmc import DMCTask
from timesead.utils.metadata import DATA_DIRECTORY
from timesead.data.transforms import DatasetSource, make_dataset_split, make_pipe_from_dict
from timesead.data.dataset import collate_fn

TASK_ID = 1
NUM_FEATURES = 768
TIME_SPLIT_WINDOW = 64


# ---------- Helper ----------
def _real_dmc_root() -> Path:
    return Path(DATA_DIRECTORY) / "dmc"


def _expected_time_split_lengths(lengths: list[int], *splits: float) -> list[list[int]]:
    total = sum(splits)
    normalized = [split / total for split in splits]
    result = [[] for _ in normalized]

    for seq_len in lengths:
        split_lengths = [int(seq_len * split) for split in normalized]
        rest = seq_len - sum(split_lengths)
        for idx in range(rest):
            split_lengths[idx] += 1

        start = 0
        for idx, current_len in enumerate(split_lengths):
            result[idx].append(current_len)
            start += current_len

    return result
# ---------- Fixture ----------
@pytest.fixture()
def real_dmc_root():
    root = _real_dmc_root()

    # check minimal folder structure
    train_dir = root / "train"
    test_dir = root / "test"

    if not train_dir.is_dir() or not test_dir.is_dir():
        pytest.skip("DMC dataset not found locally")

    return root


# ---------- Tests ----------

def test_dmc_dataset_basic_properties(real_dmc_root):
    dataset = DMCDataset(
        dataset_path=str(real_dmc_root),
        task_id=TASK_ID,
        training=True,
        preprocess=False,
    )

    assert len(dataset) > 0
    assert len(dataset) == len(dataset.seq_len)
    assert dataset.num_features == NUM_FEATURES
    assert len(dataset.get_feature_names()) == NUM_FEATURES


def test_dmc_dataset_getitem_shapes(real_dmc_root):
    dataset = DMCDataset(
        dataset_path=str(real_dmc_root),
        task_id=TASK_ID,
        training=True,
        preprocess=False,
    )

    inputs, targets = dataset[0]

    seq_len = dataset.seq_len[0]

    assert inputs[0].shape == (seq_len, NUM_FEATURES)
    assert targets[0].shape == (seq_len,)
    assert inputs[0].dtype == torch.float32
    assert targets[0].dtype == torch.int64


def test_dmc_dataset_labels(real_dmc_root):
    dataset = DMCDataset(
        dataset_path=str(real_dmc_root),
        task_id=TASK_ID,
        training=True,
        preprocess=False,
    )

    _, targets = dataset[0]

    # training should be normal only (if use_normal_only=True)
    assert torch.all(targets[0] == 0)


def test_dmc_dataset_test_mode_has_anomalies(real_dmc_root):
    dataset = DMCDataset(
        dataset_path=str(real_dmc_root),
        task_id=TASK_ID,
        training=False,
        preprocess=False,
    )

    _, targets = dataset[0]

    # test set can contain both 0 and 1
    assert set(targets[0].unique().tolist()).issubset({0, 1})


def test_dmc_lazy_loading(real_dmc_root):
    dataset = DMCDataset(
        dataset_path=str(real_dmc_root),
        task_id=TASK_ID,
        training=True,
        preprocess=False,
    )

    # before access → not loaded
    assert dataset.inputs is None
    assert dataset.targets is None

    _ = dataset[0]

    # after access → loaded
    assert dataset.inputs is not None
    assert dataset.targets is not None


def test_dmc_invalid_index(real_dmc_root):
    dataset = DMCDataset(
        dataset_path=str(real_dmc_root),
        task_id=TASK_ID,
        training=True,
        preprocess=False,
    )

    with pytest.raises(KeyError):
        _ = dataset[len(dataset) + 1]


def test_dmc_invalid_task():
    with pytest.raises(ValueError):
        DMCDataset(task_id=999)  # invalid task id


def test_dmc_time_split_and_time_first_collation_flow(real_dmc_root: Path):
    dataset = DMCDataset(
        dataset_path=str(real_dmc_root),
        task_id=TASK_ID,
        training=False,   # test mode (same as original)
        preprocess=False,
    )

    # ---- Time split ----
    train_split, val_split = list(
        make_dataset_split(dataset, 0.25, 0.75, axis="time")
    )

    expected_train_lengths, expected_val_lengths = _expected_time_split_lengths(
        dataset.seq_len, 0.25, 0.75
    )

    assert len(train_split) == len(dataset)
    assert len(val_split) == len(dataset)
    assert train_split.seq_len == expected_train_lengths
    assert val_split.seq_len == expected_val_lengths

    # ---- Pipeline (windowing only) ----
    train_pipeline = make_pipe_from_dict(
        {
            "window": {
                "class": "WindowTransform",
                "args": {"window_size": TIME_SPLIT_WINDOW},
            },
        },
        train_split,
    )

    expected_windows = sum(
        max(seq_len - TIME_SPLIT_WINDOW + 1, 0)
        for seq_len in train_split.seq_len
    )

    assert len(train_pipeline) == expected_windows
    assert train_pipeline.seq_len == TIME_SPLIT_WINDOW

    # ---- Direct sample ----
    direct_inputs, direct_targets = train_pipeline[0]

    # ---- DataLoader with time-first batching ----
    loader = torch.utils.data.DataLoader(
        train_pipeline,
        batch_size=3,
        shuffle=False,
        collate_fn=collate_fn(batch_dim=1),  # TIME-FIRST
    )

    batch_inputs, batch_targets = next(iter(loader))

    # ---- Shape checks ----
    assert batch_inputs[0].shape == (TIME_SPLIT_WINDOW, 3, dataset.num_features)
    assert batch_targets[0].shape == (TIME_SPLIT_WINDOW, 3)

    # ---- Consistency check ----
    torch.testing.assert_close(batch_inputs[0][:, 0, :], direct_inputs[0])
    torch.testing.assert_close(batch_targets[0][:, 0], direct_targets[0])