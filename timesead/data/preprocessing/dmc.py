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

from pathlib import Path

_logger = logging.getLogger(__name__)

META_DATASET_FILE = "meta_dataset.json"
STATISTICS_FILE = "train_stats.npz"
NORMAL_LABEL: bool = False
ANOMALY_LABEL: bool = True



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
    json_path: str,
    tasks: List[DMCTask],
    use_normaly_only: bool,
):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            dict_data = json.load(f)
    else:
        raise RuntimeError('Cant find meta_dataset.json in DMC folder.')
    
    train_meta_datas = []
    test_meta_datas = []
    for task in tasks:
        task_name = task.name.lower()

        test_meta_datas = dict_data[task_name]['test']
        

        
        temporary_meta_datas = dict_data[task_name]['train']
        if use_normaly_only:
            for data in temporary_meta_datas:
                if data[-1] == NORMAL_LABEL:
                    train_meta_datas.append(data)
        else:
            train_meta_datas = temporary_meta_datas
            

    return train_meta_datas, test_meta_datas

def parse_meta_data(data):
    a, b, c = zip(*data)

    # convert to lists if needed
    a, b, c = list(a), list(b), list(c)

    return a, b, c

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
    
    for task in tasks:
        task_name = task.name.lower()
        normal_task_path = os.path.join(path,"normal_features",task_name)
        with np.load(os.path.join(normal_task_path, STATISTICS_FILE)) as d:
            normal_stats = dict(d)
        final_stats = merge_stats(normal_stats, final_stats)

        if not(use_normaly_only):
            anomaly_task_path = os.path.join(path,"random_features", task_name)
            with np.load(os.path.join(anomaly_task_path, STATISTICS_FILE)) as d:
                anomaly_stats = dict(d)
            final_stats = merge_stats(anomaly_stats, final_stats)
    
    return final_stats
    
def load_features(file: str):
    data = np.load(file)
    data = data['features']
    df = pd.DataFrame(data)
    return df

def create_file_name(task_name: str, is_abnormal: bool, num_features: int):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{int(is_abnormal)}_{task_name}_{num_features}f_{ts}.csv"

def obtain_data_length_and_save_data_statistics(
        data_path: str,
        is_training: bool,
        is_anomaly: bool,
        ):

    files = glob.glob(os.path.join(data_path, "*.npz"))
    mean, min, max, n = None, None, None, 0

    file_length_pairs = []
    for file in files:
        df = load_features(file=file)
        
        if is_training:
            mean, min, max, n = update_statistics_increment(df, mean, min, max, n)
        
        p = Path(file)
        file_length_pairs.append((str(Path(*p.parts[-3:])),df.shape[0],is_anomaly))
        
    if is_training:
        save_statistic(
            out_data_dir=data_path,
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
        mean: np.ndarray,
        max: np.ndarray,
        min: np.ndarray,
        n: int):
    stats_file = os.path.join(out_data_dir, STATISTICS_FILE)
    np.savez(stats_file, mean=mean, min=min, max=max, n=n)

def construct_json(
        train_length_pairs: List[tuple[str,int]],
        test_length_pairs: List[tuple[str,int]],
        ):
    data = {
        "train": train_length_pairs,
        "test": test_length_pairs,
    }
    return data

def construct_meta_data(
        missing_tasks: List[DMCTask],
        data_dir: str
        ):
    json_file = os.path.join(data_dir, META_DATASET_FILE)
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            datas = json.load(f)
    else:
        datas = {}
    
    for task in missing_tasks:
        task_name = task.name.lower()

        normal_train, anomaly_train, normal_test, anomaly_test = get_directories_of_raw_data(
            raw_data_dir=data_dir, 
            task_name=task_name
            )

        train_length_pairs = obtain_data_length_and_save_data_statistics(
                data_path = normal_train,
                is_training = True,
                is_anomaly=False
                )
        
        train_length_pairs += obtain_data_length_and_save_data_statistics(
                data_path = anomaly_train,
                is_training = True,
                is_anomaly=True
                )
        
        test_length_pairs = obtain_data_length_and_save_data_statistics(
                data_path = normal_test,
                is_training = False,
                is_anomaly=False
                )
        
        test_length_pairs += obtain_data_length_and_save_data_statistics(
                data_path = anomaly_test,
                is_training = False,
                is_anomaly=True
                )

        data = construct_json(
            train_length_pairs=train_length_pairs,
            test_length_pairs=test_length_pairs,
        ) 
        datas[task_name] = data
    
    with open(json_file, "w") as f:
        json.dump(datas, f, indent=4)