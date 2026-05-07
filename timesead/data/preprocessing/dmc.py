from enum import Enum
import logging
import os
import json
import glob
import random
from typing import List
from datetime import datetime

import numpy as np
import pandas as pd

from .common import update_statistics_increment

from pathlib import Path

_logger = logging.getLogger(__name__)

# Constants for filenames and labels
META_DATASET_FILE = "meta_dataset.json"
NORMAL_STATISTICS_FILE = "train_normal_stats.npz"
ANOMAL_STATISTICS_FILE = "train_anomaly_stats.npz"

NORMAL_LABEL: bool = False     # normal data label
ANOMALY_LABEL: bool = True     # anomaly data label


class DMCTask(Enum):
    """
    Enum representing all DMC tasks.
    Used to select which environment/task dataset to load.
    """
    ACROBOT_SWINGUP = 0
    CHEETAH_RUN = 1
    FINGER_TURN_HARD = 2
    PENDULUM_SWINGUP = 3
    QUADRUPED_RUN = 4
    REACHER_HARD = 5
    CARTPOLE_BALANCE = 6
    CUP_CATCH = 7
    HOPPER_STAND = 8
    POINTMASS_EASY = 9
    QUADRUPED_WALK = 10
    WALKER_WALK = 11
    CARTPOLE_SWINGUP = 12
    FINGER_SPIN = 13
    MANIPULATOR_BRING_BALL = 14
    POINTMASS_HARD = 15
    REACHER_EASY = 16


def obtain_meta_data(
    json_path,
    tasks,
    use_unsupervised_training,
    use_anomalous_as_normal=False,
    *,
    shuffle_test_files: bool = False,
    shuffle_seed: int = 0,
):
    """
    Load metadata from JSON file.

    Input:
    - json_path: path to meta_dataset.json
    - tasks: list of DMCTask
    - use_unsupervised_training: whether to keep a single training class
    - use_anomalous_as_normal: whether to treat anomalous files as label 0 and normal files as label 1
    - shuffle_test_files: whether to deterministically shuffle the test file order
    - shuffle_seed: seed used for deterministic shuffling

    Output:
    - train_meta_datas: list of (file_path, length, label)
    - test_meta_datas: same format
    """

    if not os.path.exists(json_path):
        raise RuntimeError('Cant find meta_dataset.json')

    with open(json_path, "r") as f:
        dict_data = json.load(f)

    train_meta_datas = []
    test_meta_datas = []

    for task in tasks:
        task_name = task.name.lower()

        # load test metadata directly
        if use_anomalous_as_normal:
            test_meta_datas.extend(
                [file_path, length, not label]
                for file_path, length, label in dict_data[task_name]['test']
            )
        else:
            test_meta_datas.extend(dict_data[task_name]['test'])

        # load train metadata
        temporary_meta_datas = dict_data[task_name]['train']

        if use_unsupervised_training:
            train_label = ANOMALY_LABEL if use_anomalous_as_normal else NORMAL_LABEL

            for file_path, length, label in temporary_meta_datas:
                if label == train_label:
                    exposed_label = NORMAL_LABEL if use_anomalous_as_normal else label
                    train_meta_datas.append([file_path, length, exposed_label])
        else:
            train_meta_datas.extend(temporary_meta_datas)

    if shuffle_test_files and test_meta_datas:
        rng = random.Random(shuffle_seed)
        rng.shuffle(test_meta_datas)

    return train_meta_datas, test_meta_datas


def parse_meta_data(data):
    """
    Split metadata tuples into separate lists.

    Input:
    - data: list of (file_path, length, label)

    Output:
    - file_paths, lengths, labels
    """
    a, b, c = zip(*data)
    return list(a), list(b), list(c)


def merge_stats(a, b):
    """
    Merge two statistics dictionaries.

    Each dict contains:
    - mean, min, max, n (number of samples)

    Output:
    - combined statistics
    """

    # handle empty cases
    if a["n"] == 0 or a["mean"] is None:
        return b
    if b["n"] == 0 or b["mean"] is None:
        return a

    total_n = a["n"] + b["n"]

    return {
        "n": total_n,
        "mean": (a["mean"] * a["n"] + b["mean"] * b["n"]) / total_n,
        "max": np.maximum(a["max"], b["max"]),
        "min": np.minimum(a["min"], b["min"]),
    }


def get_stats(path, tasks, use_unsupervised_training, use_anomalous_as_normal=False):
    """
    Load and merge statistics across tasks.

    Input:
    - path: preprocess directory
    - tasks: list of tasks
    - use_unsupervised_training: whether training uses a single class
    - use_anomalous_as_normal: whether the anomalous class is used as that single class

    Output:
    - final_stats: combined statistics dict
    """

    final_stats = {"mean": None, "max": None, "min": None, "n": 0}

    for task in tasks:
        task_name = task.name.lower()
        task_path = os.path.join(path, task_name)

        if use_unsupervised_training:
            stats_file = ANOMAL_STATISTICS_FILE if use_anomalous_as_normal else NORMAL_STATISTICS_FILE

            with np.load(os.path.join(task_path, stats_file)) as d:
                class_stats = dict(d)

            final_stats = merge_stats(class_stats, final_stats)
        else:
            with np.load(os.path.join(task_path, NORMAL_STATISTICS_FILE)) as d:
                normal_stats = dict(d)
            final_stats = merge_stats(normal_stats, final_stats)

            with np.load(os.path.join(task_path, ANOMAL_STATISTICS_FILE)) as d:
                anomaly_stats = dict(d)
            final_stats = merge_stats(anomaly_stats, final_stats)

    return final_stats


def load_features(file):
    """
    Load .npz file and return as DataFrame.

    Output:
    - DataFrame shape (T, F)
    """
    data = np.load(file)['features']
    return pd.DataFrame(data)


def create_file_name(task_name, is_abnormal, num_features):
    """
    Generate unique filename.

    Format:
    [label]_[task]_[features]_[timestamp].csv
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{int(is_abnormal)}_{task_name}_{num_features}f_{ts}.csv"


def obtain_data_length_and_data_statistics(data_path, is_anomaly):
    """
    Process all files in a folder.

    Input:
    - data_path: directory of .npz files
    - is_anomaly: label for all files

    Output:
    - file_length_pairs: [(file_path, length, label)]
    - stats: (mean, min, max, n)
    """

    files = glob.glob(os.path.join(data_path, "*.npz"))

    mean, min, max, n = None, None, None, 0
    file_length_pairs = []

    for file in files:
        df = load_features(file)

        # update running statistics
        mean, min, max, n = update_statistics_increment(df, mean, min, max, n)

        # store relative path + length + label
        p = Path(file)
        file_length_pairs.append(
            (str(Path(*p.parts[-3:])), df.shape[0], is_anomaly)
        )

    return file_length_pairs, (mean, min, max, n)


def get_directories_of_raw_data(raw_data_dir, task_name):
    """
    Get all relevant directories for a task.

    Output:
    - normal_train_dir
    - anomaly_train_dir
    - normal_test_dir
    - anomaly_test_dir
    """

    return (
        os.path.join(raw_data_dir, 'train', "normal_features", task_name),
        os.path.join(raw_data_dir, 'train', "random_features", task_name),
        os.path.join(raw_data_dir, 'test', "normal_features", task_name),
        os.path.join(raw_data_dir, 'test', "random_features", task_name),
    )


def construct_json(train_length_pairs, test_length_pairs):
    """
    Build JSON structure for one task.

    Output:
    {
        "train": [...],
        "test": [...]
    }
    """
    return {
        "train": train_length_pairs,
        "test": test_length_pairs,
    }


def construct_meta_data(missing_tasks, data_dir, save_dir):
    """
    Main preprocessing function.

    Goal:
    - Scan raw dataset
    - Compute statistics
    - Build metadata JSON
    - Save everything to disk

    Steps per task:
    1. Locate raw data folders
    2. Compute stats for normal + anomaly
    3. Save stats (.npz)
    4. Build metadata (file paths, lengths, labels)
    5. Save to JSON
    """

    os.makedirs(save_dir, exist_ok=True)

    json_file = os.path.join(save_dir, META_DATASET_FILE)

    # load existing metadata if exists
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            datas = json.load(f)
    else:
        datas = {}

    for task in missing_tasks:
        task_name = task.name.lower()

        # get all directories
        normal_train, anomaly_train, normal_test, anomaly_test = \
            get_directories_of_raw_data(data_dir, task_name)

        task_path = os.path.join(save_dir, task_name)
        os.makedirs(task_path, exist_ok=True)

        # ---- TRAIN NORMAL ----
        train_length_pairs, stats = obtain_data_length_and_data_statistics(
            normal_train, is_anomaly=False
        )

        mean, min, max, n = stats
        np.savez(os.path.join(task_path, NORMAL_STATISTICS_FILE),
                 mean=mean, min=min, max=max, n=n)

        # ---- TRAIN ANOMALY ----
        temp_pairs, stats = obtain_data_length_and_data_statistics(
            anomaly_train, is_anomaly=True
        )

        mean, min, max, n = stats
        np.savez(os.path.join(task_path, ANOMAL_STATISTICS_FILE),
                 mean=mean, min=min, max=max, n=n)

        train_length_pairs += temp_pairs

        # ---- TEST DATA ----
        test_length_pairs, _ = obtain_data_length_and_data_statistics(
            normal_test, is_anomaly=False
        )

        temp_pairs, _ = obtain_data_length_and_data_statistics(
            anomaly_test, is_anomaly=True
        )

        test_length_pairs += temp_pairs

        # build JSON entry
        datas[task_name] = construct_json(
            train_length_pairs,
            test_length_pairs
        )

    # save final metadata file
    with open(json_file, "w") as f:
        json.dump(datas, f, indent=4)
