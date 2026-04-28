import json
import random

from timesead.data.preprocessing.dmc import DMCTask, obtain_meta_data


def test_obtain_meta_data_can_deterministically_shuffle_test_files(tmp_path):
    # Build a synthetic meta_dataset.json that mimics the real structure.
    # JSON roundtrips tuples as lists, so keep entries list-shaped.
    test_entries = [
        [f"test/file_{idx:02d}.npz", 100 + idx, bool(idx % 2)]
        for idx in range(12)
    ]
    meta = {
        "cheetah_run": {
            "train": [
                ["train/normal_0.npz", 50, False],
                ["train/anom_0.npz", 50, True],
            ],
            "test": test_entries,
        }
    }
    meta_path = tmp_path / "meta_dataset.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    _, test_unshuffled = obtain_meta_data(
        meta_path,
        tasks=[DMCTask.CHEETAH_RUN],
        use_normaly_only=False,
        shuffle_test_files=False,
    )
    assert test_unshuffled == test_entries

    expected_seed0 = list(test_entries)
    random.Random(0).shuffle(expected_seed0)
    _, test_shuffled_seed0 = obtain_meta_data(
        meta_path,
        tasks=[DMCTask.CHEETAH_RUN],
        use_normaly_only=False,
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
        use_normaly_only=False,
        shuffle_test_files=True,
        shuffle_seed=1,
    )
    assert test_shuffled_seed1 == expected_seed1
