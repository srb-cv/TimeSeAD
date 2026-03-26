from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from timesead.data.dataset import collate_fn
from timesead.data.tep_dataset import TEPDataset
from timesead.data.transforms import DatasetSource, make_dataset_split, make_pipe_from_dict


TRAIN_SEQ_LEN = 500
TEST_SEQ_LEN = 960
NUM_FEATURES = 52


def _build_tep_frame(seq_len: int, runs: list[int], fault_number: int) -> pd.DataFrame:
    feature_names = TEPDataset.get_feature_names()
    rows = []

    for run in runs:
        for sample in range(1, seq_len + 1):
            row = {
                "simulationRun": run,
                "sample": sample,
                "faultNumber": fault_number,
            }
            for feature_idx, feature_name in enumerate(feature_names):
                row[feature_name] = fault_number * 1000 + run * 100 + sample + feature_idx / 100
            rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_tep_dir(tmp_path: Path) -> Path:
    root = tmp_path / "TEP_harvard"
    processed_dir = root / "processed"
    processed_dir.mkdir(parents=True)

    np.savez(
        processed_dir / "TEP_FaultFree_Training_stats.npz",
        mean=np.zeros(NUM_FEATURES, dtype=np.float32),
        std=np.ones(NUM_FEATURES, dtype=np.float32),
    )

    runs = [1, 2]
    _build_tep_frame(TRAIN_SEQ_LEN, runs, 0).to_csv(processed_dir / "TEP_FaultFree_Training.csv", index=False)
    _build_tep_frame(TRAIN_SEQ_LEN, runs, 1).to_csv(processed_dir / "TEP_Faulty_Training_01.csv", index=False)
    _build_tep_frame(TEST_SEQ_LEN, runs, 0).to_csv(processed_dir / "TEP_FaultFree_Testing.csv", index=False)
    _build_tep_frame(TEST_SEQ_LEN, runs, 1).to_csv(processed_dir / "TEP_Faulty_Testing_01.csv", index=False)

    return root


def test_tep_dataset_returns_full_sequences_per_fault_and_run(synthetic_tep_dir: Path):
    dataset = TEPDataset(path=str(synthetic_tep_dir), faults=[0, 1], runs=[1, 2], training=True, preprocess=False)

    assert len(dataset) == 4
    assert dataset.seq_len == TRAIN_SEQ_LEN
    assert dataset.num_features == NUM_FEATURES

    normal_inputs, normal_targets = dataset[0]
    faulty_inputs, faulty_targets = dataset[2]

    assert normal_inputs[0].shape == (TRAIN_SEQ_LEN, NUM_FEATURES)
    assert normal_targets[0].shape == (TRAIN_SEQ_LEN,)
    assert faulty_inputs[0].shape == (TRAIN_SEQ_LEN, NUM_FEATURES)
    assert faulty_targets[0].shape == (TRAIN_SEQ_LEN,)

    assert torch.count_nonzero(normal_targets[0]) == 0
    assert torch.all(faulty_targets[0][:20] == 0)
    assert torch.all(faulty_targets[0][20:] == 1)
    assert faulty_inputs[0][0, 0].item() != normal_inputs[0][0, 0].item()


def test_tep_default_pipeline_converts_fault_labels_to_binary_targets(synthetic_tep_dir: Path):
    dataset = TEPDataset(path=str(synthetic_tep_dir), faults=[0, 1], runs=[1], training=False, preprocess=False)

    pipeline_dataset = make_pipe_from_dict(dataset.get_default_pipeline(), DatasetSource(dataset))
    inputs, targets = pipeline_dataset[1]

    assert inputs[0].shape == (TEST_SEQ_LEN, NUM_FEATURES)
    assert targets[0].shape == (TEST_SEQ_LEN,)
    assert set(targets[0].unique().tolist()) == {0, 1}
    assert torch.all(targets[0][:160] == 0)
    assert torch.all(targets[0][160:] == 1)


def test_tep_batch_split_window_and_reconstruction_flow(synthetic_tep_dir: Path):
    dataset = TEPDataset(path=str(synthetic_tep_dir), faults=[0, 1], runs=[1, 2], training=True, preprocess=False)
    train_split, val_split = list(make_dataset_split(dataset, 0.5, 0.5, axis="batch"))

    assert len(train_split) == 2
    assert len(val_split) == 2
    assert train_split.seq_len == TRAIN_SEQ_LEN
    assert val_split.seq_len == TRAIN_SEQ_LEN

    train_pipeline = make_pipe_from_dict(
        {
            "window": {"class": "WindowTransform", "args": {"window_size": 50}},
            "reconstruction": {"class": "ReconstructionTargetTransform", "args": {"replace_labels": True}},
        },
        train_split,
    )

    assert len(train_pipeline) == 2 * (TRAIN_SEQ_LEN - 50 + 1)
    assert train_pipeline.seq_len == 50

    loader = torch.utils.data.DataLoader(
        train_pipeline,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn(batch_dim=0),
    )
    batch_inputs, batch_targets = next(iter(loader))

    assert batch_inputs[0].shape == (4, 50, NUM_FEATURES)
    assert batch_targets[0].shape == (4, 50, NUM_FEATURES)
    torch.testing.assert_close(batch_inputs[0], batch_targets[0])


def test_tep_time_split_and_time_first_collation_flow(synthetic_tep_dir: Path):
    dataset = TEPDataset(path=str(synthetic_tep_dir), faults=[0, 1], runs=[1, 2], training=False, preprocess=False)
    train_split, val_split = list(make_dataset_split(dataset, 0.25, 0.75, axis="time"))

    assert len(train_split) == 4
    assert len(val_split) == 4
    assert train_split.seq_len == [240, 240, 240, 240]
    assert val_split.seq_len == [720, 720, 720, 720]

    train_pipeline = make_pipe_from_dict(
        {
            "window": {"class": "WindowTransform", "args": {"window_size": 100}},
        },
        train_split,
    )

    assert len(train_pipeline) == 4 * (240 - 100 + 1)
    assert train_pipeline.seq_len == 100

    direct_inputs, direct_targets = train_pipeline[0]

    loader = torch.utils.data.DataLoader(
        train_pipeline,
        batch_size=3,
        shuffle=False,
        collate_fn=collate_fn(batch_dim=1),
    )
    batch_inputs, batch_targets = next(iter(loader))

    assert batch_inputs[0].shape == (100, 3, NUM_FEATURES)
    assert batch_targets[0].shape == (100, 3)
    torch.testing.assert_close(batch_inputs[0][:, 0, :], direct_inputs[0])
    torch.testing.assert_close(batch_targets[0][:, 0], direct_targets[0])
