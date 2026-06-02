import json
import random

from timesead.data.preprocessing.dmc import DMCTask, obtain_meta_data


def _write_meta_dataset(tmp_path, train_entries, test_entries=None):
    meta = {
        "cheetah_run": {
            "train": train_entries,
            "test": test_entries or [],
        }
    }
    meta_path = tmp_path / "meta_dataset.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return meta_path


def test_obtain_meta_data_selects_normal_files_for_unsupervised_training(tmp_path):
    train_entries = [
        ["normal_features/cheetah_run/normal_0.npz", 50, False],
        ["random_features/cheetah_run/anom_0.npz", 60, True],
    ]

    train_meta, _ = obtain_meta_data(
        _write_meta_dataset(tmp_path, train_entries),
        tasks=[DMCTask.CHEETAH_RUN],
        use_unsupervised_training=True,
    )

    assert train_meta == [["normal_features/cheetah_run/normal_0.npz", 50, False]]


def test_obtain_meta_data_can_use_anomalous_files_as_normal(tmp_path):
    train_entries = [
        ["normal_features/cheetah_run/normal_0.npz", 50, False],
        ["random_features/cheetah_run/anom_0.npz", 60, True],
    ]

    train_meta, _ = obtain_meta_data(
        _write_meta_dataset(tmp_path, train_entries),
        tasks=[DMCTask.CHEETAH_RUN],
        use_unsupervised_training=True,
        use_anomalous_as_normal=True,
    )

    assert train_meta == [["random_features/cheetah_run/anom_0.npz", 60, False]]


def test_obtain_meta_data_reverses_test_labels_when_using_anomalous_as_normal(tmp_path):
    train_entries = [
        ["normal_features/cheetah_run/normal_0.npz", 50, False],
        ["random_features/cheetah_run/anom_0.npz", 60, True],
    ]
    test_entries = [
        ["normal_features/cheetah_run/normal_test.npz", 70, False],
        ["random_features/cheetah_run/anom_test.npz", 80, True],
    ]

    _, test_meta = obtain_meta_data(
        _write_meta_dataset(tmp_path, train_entries, test_entries),
        tasks=[DMCTask.CHEETAH_RUN],
        use_unsupervised_training=True,
        use_anomalous_as_normal=True,
    )

    assert test_meta == [
        ["normal_features/cheetah_run/normal_test.npz", 70, True],
        ["random_features/cheetah_run/anom_test.npz", 80, False],
    ]


def test_obtain_meta_data_keeps_all_training_files_for_supervised_training(tmp_path):
    train_entries = [
        ["normal_features/cheetah_run/normal_0.npz", 50, False],
        ["random_features/cheetah_run/anom_0.npz", 60, True],
    ]

    train_meta, _ = obtain_meta_data(
        _write_meta_dataset(tmp_path, train_entries),
        tasks=[DMCTask.CHEETAH_RUN],
        use_unsupervised_training=False,
    )

    assert train_meta == train_entries


def test_obtain_meta_data_can_deterministically_shuffle_train_files(tmp_path):
    # Build a synthetic meta_dataset.json that mimics the real structure.
    # JSON roundtrips tuples as lists, so keep entries list-shaped.
    train_entries = [
        [f"train/file_{idx:02d}.npz", 100 + idx, bool(idx % 2)]
        for idx in range(12)
    ]
    meta_path = _write_meta_dataset(tmp_path, train_entries=train_entries)

    train_unshuffled, _ = obtain_meta_data(
        meta_path,
        tasks=[DMCTask.CHEETAH_RUN],
        use_unsupervised_training=False,
        shuffle_train_files=False,
    )
    assert train_unshuffled == train_entries

    expected_seed0 = list(train_entries)
    random.Random(0).shuffle(expected_seed0)
    train_shuffled_seed0, _ = obtain_meta_data(
        meta_path,
        tasks=[DMCTask.CHEETAH_RUN],
        use_unsupervised_training=False,
        shuffle_train_files=True,
        shuffle_seed=0,
    )
    assert train_shuffled_seed0 == expected_seed0
    assert sorted(train_shuffled_seed0) == sorted(train_entries)


def test_obtain_meta_data_can_deterministically_shuffle_test_files(tmp_path):
    # Build a synthetic meta_dataset.json that mimics the real structure.
    # JSON roundtrips tuples as lists, so keep entries list-shaped.
    test_entries = [
        [f"test/file_{idx:02d}.npz", 100 + idx, bool(idx % 2)]
        for idx in range(12)
    ]
    meta_path = _write_meta_dataset(
        tmp_path,
        train_entries=[
            ["train/normal_0.npz", 50, False],
            ["train/anom_0.npz", 50, True],
        ],
        test_entries=test_entries,
    )

    _, test_unshuffled = obtain_meta_data(
        meta_path,
        tasks=[DMCTask.CHEETAH_RUN],
        use_unsupervised_training=False,
        shuffle_test_files=False,
    )
    assert test_unshuffled == test_entries

    expected_seed0 = list(test_entries)
    random.Random(0).shuffle(expected_seed0)
    _, test_shuffled_seed0 = obtain_meta_data(
        meta_path,
        tasks=[DMCTask.CHEETAH_RUN],
        use_unsupervised_training=False,
        shuffle_test_files=True,
        shuffle_seed=0,
    )
    assert test_shuffled_seed0 == expected_seed0
    assert sorted(test_shuffled_seed0) == sorted(test_entries)

    expected_seed1 = list(test_entries)
    random.Random(1).shuffle(expected_seed1)
    _, test_shuffled_seed1 = obtain_meta_data(
        meta_path,
        tasks=[DMCTask.CHEETAH_RUN],
        use_unsupervised_training=False,
        shuffle_test_files=True,
        shuffle_seed=1,
    )
    assert test_shuffled_seed1 == expected_seed1
