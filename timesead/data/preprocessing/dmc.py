from enum import Enum
import logging
import os
import json
import glob
from typing import List
from datetime import datetime

import numpy as np
import pandas as pd

from .common import update_statistics_increment

_logger = logging.getLogger(__name__)


class DMCTask(Enum):
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
    path: str,
    tasks: List[DMCTask],
    use_normaly_only: bool,
):
    json_file = os.path.join(path, 'meta_dataset.json')
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            dict_data = json.load(f)
    else:
        raise RuntimeError('Cant find meta_dataset.json in DMC folder.')
    train_files = []
    train_lengths = []

    test_files = []
    test_lengths = []

    for task in tasks:
        task_name = task.name.lower()

        files, lengths = zip(*dict_data[task_name]['test'])
        files = list(files)
        lengths = list(lengths)
        test_files += [os.path.join(task_name, file) for file in files]
        test_lengths += lengths

        files, lengths = zip(*dict_data[task_name]['train']['normal'])
        files = list(files)
        lengths = list(lengths)
        train_files += [os.path.join(task_name, file) for file in files]
        train_lengths += lengths

        if not(use_normaly_only):
            files, lengths = zip(*dict_data[task_name]['train']['anomaly'])
            files = list(files)
            lengths = list(lengths)
            train_files += [os.path.join(task_name, file) for file in files]
            train_lengths += lengths

    return train_files, train_lengths, test_files, test_lengths

def merge_stats(a, b):
    # If one is empty → return the other
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

def get_stats(
        path:str,
        tasks: List[DMCTask],
        use_normaly_only: bool,
        ):
    final_stats = {
        "mean": None,
        "max": None,
        "min": None,
        "n": 0,
    }
    
    for task in tasks[1:]:
        task_path = os.path.join(path, task.name.lower())
        with np.load(os.path.join(task_path, 'train_stats_normal.npz')) as d:
            normal_stats = dict(d)
        final_stats = merge_stats(normal_stats, final_stats)

        if not(use_normaly_only):
            with np.load(os.path.join(task_path, 'train_stats_anomaly.npz')) as d:
                anomaly_stats = dict(d)
            final_stats = merge_stats(anomaly_stats, final_stats)
    
    return final_stats
    
def load_preprocessed_features(file: str):
    data = np.load(file)
    data = data['features']
    df = pd.DataFrame(data)

    column_names = [f"feature_{i}" for i in range(df.shape[1])]
    df.columns = column_names
    return df

def create_file_name(task_name: str, is_abnormal: bool, num_features: int):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{int(is_abnormal)}_{task_name}_{num_features}f_{ts}.csv"

def create_new_dataset_with_label(
        raw_path: str,
        output_path: str,
        is_training: bool,
        task_name: str,
        is_abnormal: bool,
        ):

    files = glob.glob(os.path.join(raw_path, "*.npz"))
    mean, min, max, n = None, None, None, 0

    file_length_pairs = []
    for file in files:
        df = load_preprocessed_features(
            file=file
            )
        
        if is_training:
            mean, min, max, n = update_statistics_increment(df, mean, min, max, n)
        
        if is_abnormal:
            new_col = np.ones((df.shape[0], 1))
        else:
            new_col = np.zeros((df.shape[0], 1))
        df["Anomaly"] = new_col
        
        file_name = create_file_name(
            task_name=task_name,
            is_abnormal=is_abnormal,
            num_features=df.shape[1]
            )
        
        df.to_csv(os.path.join(output_path, file_name), index=False)
        file_length_pairs.append((file_name,df.shape[0]))
        
    if is_training:
        save_statistic(
            out_data_dir=output_path,
            feature_type= "anomaly" if is_abnormal else "normal",
            max=max,
            mean=mean,
            min=min,
            n=n,
        )
    return file_length_pairs

def get_directories_of_raw_data(raw_data_dir: str, task_name: str):
    normal_train_dir = os.path.join(raw_data_dir, 'train', "normal_features",task_name)
    anomaly_train_dir = os.path.join(raw_data_dir, 'train', "random_features",task_name)

    normal_test_dir = os.path.join(raw_data_dir, 'test', "normal_features",task_name)
    anomaly_test_dir = os.path.join(raw_data_dir, 'test', "random_features",task_name)

    return (normal_train_dir, anomaly_train_dir, normal_test_dir, anomaly_test_dir)

def save_statistic(
        out_data_dir: str,
        feature_type: str,
        mean: np.ndarray,
        max: np.ndarray,
        min: np.ndarray,
        n: int):
    stats_file = os.path.join(out_data_dir, f'train_stats_{feature_type}.npz')
    np.savez(stats_file, mean=mean, min=min, max=max, n=n)

def construct_json(
        normal_train_length_pairs: List[tuple[str,int]],
        anomaly_train_length_pairs: List[tuple[str,int]],
        test_length_pairs: List[tuple[str,int]],
        task_name: str,
        ):
    data = {
        "train":{
            "normal": normal_train_length_pairs,
            "anomaly": anomaly_train_length_pairs,
        },
        "test": test_length_pairs
    }
    return data

def preprocess_dmc_data(missing_tasks: List[DMCTask], out_data_dir: str, raw_data_dir: str):
    train_out_data_dir = os.path.join(out_data_dir, 'train')
    os.makedirs(train_out_data_dir, exist_ok=True)
    test_out_data_dir = os.path.join(out_data_dir, 'test')
    os.makedirs(test_out_data_dir, exist_ok=True)
    
    json_file = os.path.join(out_data_dir, 'meta_dataset.json')
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            datas = json.load(f)
    else:
        datas = {}
    
    for task in missing_tasks:
        task_name = task.name.lower()
        train_task_path = os.path.join(train_out_data_dir, task_name)
        test_task_path = os.path.join(test_out_data_dir, task_name)
        os.makedirs(train_task_path, exist_ok=True)
        os.makedirs(test_task_path, exist_ok=True)

        normal_train, anomaly_train, normal_test, anomaly_test = get_directories_of_raw_data(
            raw_data_dir=raw_data_dir, 
            task_name=task.name.lower()
            )


        normal_train_length_pairs = create_new_dataset_with_label(
            raw_path=normal_train,
            output_path=train_task_path,
            is_training=True,
            task_name=task_name,
            is_abnormal=False,
        )

        anomaly_train_length_pairs = create_new_dataset_with_label(
            raw_path=anomaly_train,
            output_path=train_task_path,
            is_training=True,
            task_name=task_name,
            is_abnormal=True,
        )

        test_length_pairs = create_new_dataset_with_label(
            raw_path=normal_test,
            output_path=test_task_path,
            is_training=False,
            task_name=task_name,
            is_abnormal=False,
        )

        test_length_pairs += create_new_dataset_with_label(
            raw_path=anomaly_test,
            output_path=test_task_path,
            is_training=False,
            task_name=task_name,
            is_abnormal=True,
        )
        data = construct_json(
            normal_train_length_pairs = normal_train_length_pairs,
            anomaly_train_length_pairs=anomaly_train_length_pairs,
            test_length_pairs=test_length_pairs,
            task_name=task_name
        ) 
        datas[task_name] = data
    
    with open(json_file, "w") as f:
        json.dump(datas, f, indent=4)