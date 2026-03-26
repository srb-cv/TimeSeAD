from pathlib import Path
import os

import pytest
from pytest import MonkeyPatch
import torch

from timesead.data.dataset import collate_fn
from timesead.data.exathlon_dataset import ExathlonDataset
from timesead.data.transforms import DatasetSource, make_dataset_split, make_pipe_from_dict
from timesead.utils.metadata import DATA_DIRECTORY


APP_ID = 1
NUM_FEATURES = 19
TRAIN_WINDOW = 128
TIME_SPLIT_WINDOW = 64


def _real_exathlon_root() -> Path:
    return Path(DATA_DIRECTORY) / "exathlon"


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


@pytest.fixture()
def real_exathlon_root() -> Path:
    root = _real_exathlon_root()
    processed_dir = root / "data" / "processed"
    if not processed_dir.is_dir():
        pytest.skip(
            f"Real Exathlon dataset not available at {processed_dir}. "
            "These tests require the locally prepared dataset."
        )

    return root


def test_exathlon_dataset_returns_full_sequences_per_trace(real_exathlon_root: Path):
    dataset = ExathlonDataset(
        dataset_path=str(real_exathlon_root),
        app_id=APP_ID,
        training=True,
        download=False,
        preprocess=False,
    )

    assert len(dataset) == len(dataset.seq_len)
    assert len(dataset) > 1
    assert dataset.num_features == NUM_FEATURES
    assert len(dataset.get_feature_names()) == NUM_FEATURES

    first_inputs, first_targets = dataset[0]
    second_inputs, second_targets = dataset[1]

    assert first_inputs[0].shape == (dataset.seq_len[0], NUM_FEATURES)
    assert first_targets[0].shape == (dataset.seq_len[0],)
    assert second_inputs[0].shape == (dataset.seq_len[1], NUM_FEATURES)
    assert second_targets[0].shape == (dataset.seq_len[1],)

    assert first_inputs[0].dtype == torch.float32
    assert first_targets[0].dtype == torch.int64
    assert torch.count_nonzero(first_targets[0]) == 0
    assert torch.count_nonzero(second_targets[0]) == 0
    assert first_inputs[0].shape[0] != second_inputs[0].shape[0]


def test_exathlon_test_dataset_returns_labeled_full_sequences(real_exathlon_root: Path):
    dataset = ExathlonDataset(
        dataset_path=str(real_exathlon_root),
        app_id=APP_ID,
        training=False,
        download=False,
        preprocess=False,
    )

    assert len(dataset) == len(dataset.seq_len)
    assert len(dataset) > 0
    assert dataset.num_features == NUM_FEATURES

    inputs, targets = dataset[0]

    assert inputs[0].shape == (dataset.seq_len[0], NUM_FEATURES)
    assert targets[0].shape == (dataset.seq_len[0],)
    assert inputs[0].dtype == torch.float32
    assert targets[0].dtype == torch.int64
    assert set(targets[0].unique().tolist()).issubset({0, 1})


def test_exathlon_batch_split_window_and_reconstruction_flow(real_exathlon_root: Path):
    dataset = ExathlonDataset(
        dataset_path=str(real_exathlon_root),
        app_id=APP_ID,
        training=True,
        download=False,
        preprocess=False,
    )
    train_split, val_split = list(make_dataset_split(dataset, 0.5, 0.5, axis="batch"))

    assert len(train_split) + len(val_split) == len(dataset)
    assert train_split.seq_len == dataset.seq_len[:len(train_split)]
    assert val_split.seq_len == dataset.seq_len[len(train_split):]

    train_pipeline = make_pipe_from_dict(
        {
            "window": {"class": "WindowTransform", "args": {"window_size": TRAIN_WINDOW}},
            "reconstruction": {"class": "ReconstructionTargetTransform", "args": {"replace_labels": True}},
        },
        train_split,
    )

    expected_train_windows = sum(max(seq_len - TRAIN_WINDOW + 1, 0) for seq_len in train_split.seq_len)

    assert len(train_pipeline) == expected_train_windows
    assert train_pipeline.seq_len == TRAIN_WINDOW

    loader = torch.utils.data.DataLoader(
        train_pipeline,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn(batch_dim=0),
    )
    batch_inputs, batch_targets = next(iter(loader))

    assert batch_inputs[0].shape == (4, TRAIN_WINDOW, NUM_FEATURES)
    assert batch_targets[0].shape == (4, TRAIN_WINDOW, NUM_FEATURES)
    torch.testing.assert_close(batch_inputs[0], batch_targets[0])


def test_exathlon_time_split_and_time_first_collation_flow(real_exathlon_root: Path):
    dataset = ExathlonDataset(
        dataset_path=str(real_exathlon_root),
        app_id=APP_ID,
        training=False,
        download=False,
        preprocess=False,
    )
    train_split, val_split = list(make_dataset_split(dataset, 0.25, 0.75, axis="time"))
    expected_train_lengths, expected_val_lengths = _expected_time_split_lengths(dataset.seq_len, 0.25, 0.75)

    assert len(train_split) == len(dataset)
    assert len(val_split) == len(dataset)
    assert train_split.seq_len == expected_train_lengths
    assert val_split.seq_len == expected_val_lengths

    train_pipeline = make_pipe_from_dict(
        {
            "window": {"class": "WindowTransform", "args": {"window_size": TIME_SPLIT_WINDOW}},
        },
        train_split,
    )

    expected_windows = sum(max(seq_len - TIME_SPLIT_WINDOW + 1, 0) for seq_len in train_split.seq_len)

    assert len(train_pipeline) == expected_windows
    assert train_pipeline.seq_len == TIME_SPLIT_WINDOW

    direct_inputs, direct_targets = train_pipeline[0]

    loader = torch.utils.data.DataLoader(
        train_pipeline,
        batch_size=3,
        shuffle=False,
        collate_fn=collate_fn(batch_dim=1),
    )
    batch_inputs, batch_targets = next(iter(loader))

    assert batch_inputs[0].shape == (TIME_SPLIT_WINDOW, 3, NUM_FEATURES)
    assert batch_targets[0].shape == (TIME_SPLIT_WINDOW, 3)
    torch.testing.assert_close(batch_inputs[0][:, 0, :], direct_inputs[0])
    torch.testing.assert_close(batch_targets[0][:, 0], direct_targets[0])


def test_exathlon_constructor_can_download_and_prepare_real_data(tmp_path: Path):
    mp = MonkeyPatch()
    mp.setenv("RUN_EXATHLON_DOWNLOAD_TEST", "1")
    if os.environ.get("RUN_EXATHLON_DOWNLOAD_TEST") != "1":
        pytest.skip(
            "Set RUN_EXATHLON_DOWNLOAD_TEST=1 to run the Exathlon download test. "
            "This test downloads and preprocesses the real dataset."
        )

    dataset_root = Path("./data/exathlon")
    dataset = ExathlonDataset(
        dataset_path=str(dataset_root),
        app_id=APP_ID,
        training=True,
        download=False,
        preprocess=True,
    )

    assert (dataset_root / "data").is_dir()
    assert (dataset_root / "data" / "processed").is_dir()
    assert len(dataset) == len(dataset.seq_len)
    assert len(dataset) > 1
    assert dataset.num_features == NUM_FEATURES

    inputs, targets = dataset[0]

    assert inputs[0].shape == (dataset.seq_len[0], NUM_FEATURES)
    assert targets[0].shape == (dataset.seq_len[0],)
    assert inputs[0].dtype == torch.float32
    assert targets[0].dtype == torch.int64
