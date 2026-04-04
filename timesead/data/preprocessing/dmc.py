from enum import Enum
import logging
import os
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

def load_preprocessed_features(file: str, is_abnormal: bool):
    data = np.load(file)
    data = data['features']
    
    if is_abnormal:
        new_col = np.ones((data.shape[0], 1))
    else:
        new_col = np.zeros((data.shape[0], 1))

    data = np.hstack((data, new_col))

    df = pd.DataFrame(data)

    column_names = [f"feature_{i}" for i in range(df.shape[1] - 1)]
    column_names.append("Anomaly")
    df.columns = column_names
    return df


def create_file_name(task_name: str, is_abnormal: bool, num_features: int):
    ts = datetime.now().strftime("%Y%m%d_%H%M")
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
    for file in files:
        df = load_preprocessed_features(
            file=file,
            is_abnormal=is_abnormal
            )
        
        file_name = create_file_name(
            task_name=task_name,
            is_abnormal=is_abnormal,
            num_features=df.shape[1]
            )
        
        df.to_csv(os.path.join(output_path, file_name), index=False)

        if is_training:
            mean, min, max, n = update_statistics_increment(df, mean, min, max, n)
        
    if is_training:
        save_statistic(
            out_data_dir=output_path,
            feature_type= "anomaly" if is_abnormal else "normal",
            max=max,
            mean=mean,
            min=min,
        )
    return

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
        min: np.ndarray,):
    stats_file = os.path.join(out_data_dir, 'train', f'train_stats_{feature_type}.npz')
    np.savez(stats_file, mean=mean, min=min, max=max)

def preprocess_dmc_data(missing_tasks: List[DMCTask], out_data_dir: str, raw_data_dir: str):
    train_out_data_dir = os.path.join(out_data_dir, 'train')
    os.makedirs(train_out_data_dir, exist_ok=True)
    test_out_data_dir = os.path.join(out_data_dir, 'test')
    os.makedirs(test_out_data_dir, exist_ok=True)

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
        
        create_new_dataset_with_label(
            raw_path=normal_train,
            output_path=train_task_path,
            is_training=True,
            task_name=task_name,
            is_abnormal=False,
        )

        create_new_dataset_with_label(
            raw_path=anomaly_train,
            output_path=train_task_path,
            is_training=True,
            task_name=task_name,
            is_abnormal=True,
        )

        create_new_dataset_with_label(
            raw_path=normal_test,
            output_path=test_task_path,
            is_training=False,
            task_name=task_name,
            is_abnormal=False,
        )

        create_new_dataset_with_label(
            raw_path=anomaly_test,
            output_path=test_task_path,
            is_training=False,
            task_name=task_name,
            is_abnormal=True,
        )        


    return