import os
import json
from pathlib import Path

import pytest
import torch
import numpy as np

from timesead.data.dmc_dataset import DMCDataset, _default_preprocess_path
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


def _write_feature_file(path: Path, data: np.ndarray, feature_key: str = "features"):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{feature_key: data.astype(np.float32)})


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

    preprocess_meta = Path(_default_preprocess_path("videomae")) / "meta_dataset.json"
    if not preprocess_meta.is_file():
        pytest.skip("DMC videomae preprocess cache not found locally")

    return root


# ---------- Tests ----------

def test_dmc_default_preprocess_path_is_feature_set_specific():
    assert Path(_default_preprocess_path("vqgan")).parts[-4:] == (
        "data",
        "dmc",
        "preprocess",
        "vqgan",
    )


def test_dmc_dataset_supports_configurable_feature_dirs_and_infers_feature_size(tmp_path):
    dataset_root = tmp_path / "dmc"
    preprocess_path = tmp_path / "preprocess" / "vqgan"
    feature_key = "embeddings"

    _write_feature_file(
        dataset_root / "train" / "vqgan_normal" / "cheetah_run" / "normal_train.npz",
        np.arange(20).reshape(4, 5),
        feature_key=feature_key,
    )
    _write_feature_file(
        dataset_root / "train" / "vqgan_random" / "cheetah_run" / "anomaly_train.npz",
        np.arange(15).reshape(3, 5),
        feature_key=feature_key,
    )
    _write_feature_file(
        dataset_root / "test" / "vqgan_normal" / "cheetah_run" / "normal_test.npz",
        np.arange(10).reshape(2, 5),
        feature_key=feature_key,
    )
    _write_feature_file(
        dataset_root / "test" / "vqgan_random" / "cheetah_run" / "anomaly_test.npz",
        np.arange(10, 20).reshape(2, 5),
        feature_key=feature_key,
    )

    dataset = DMCDataset(
        dataset_path=str(dataset_root),
        task_id=2,
        training=True,
        standardize=False,
        preprocess=True,
        feature_set="vqgan",
        normal_feature_dir="vqgan_normal",
        anomaly_feature_dir="vqgan_random",
        preprocess_path=str(preprocess_path),
        feature_key=feature_key,
    )

    assert dataset.num_features == 5
    assert len(dataset.get_feature_names()) == 5
    assert dataset.train_files == ["vqgan_normal/cheetah_run/normal_train.npz"]

    inputs, targets = dataset[0]
    assert inputs[0].shape == (4, 5)
    assert targets[0].tolist() == [0, 0, 0, 0]

    meta = json.loads((preprocess_path / "meta_dataset.json").read_text(encoding="utf-8"))
    assert meta["_metadata"] == {
        "feature_set": "vqgan",
        "normal_feature_dir": "vqgan_normal",
        "anomaly_feature_dir": "vqgan_random",
        "feature_key": feature_key,
    }
    assert meta["cheetah_run"]["test"] == [
        ["vqgan_normal/cheetah_run/normal_test.npz", 2, False],
        ["vqgan_random/cheetah_run/anomaly_test.npz", 2, True],
    ]

    test_dataset = DMCDataset(
        dataset_path=str(dataset_root),
        task_id=2,
        training=False,
        standardize=False,
        preprocess=False,
        feature_set="vqgan",
        normal_feature_dir="vqgan_normal",
        anomaly_feature_dir="vqgan_random",
        preprocess_path=str(preprocess_path),
        feature_key=feature_key,
    )
    assert test_dataset.num_features == 5
    assert test_dataset.test_files == [
        "vqgan_normal/cheetah_run/normal_test.npz",
        "vqgan_random/cheetah_run/anomaly_test.npz",
    ]


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

    # unsupervised training exposes one training class as normal
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
