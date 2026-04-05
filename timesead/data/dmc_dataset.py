import functools
import os
from typing import Tuple, Optional, Union, Callable, Dict, Any, List
import tempfile
import subprocess
import shutil
import logging

import numpy as np
import pandas as pd
import torch

from timesead.data.dataset import BaseTSDataset
from timesead.data.preprocessing import minmax_scaler
from timesead.utils.metadata import DATA_DIRECTORY
from timesead.data.preprocessing.dmc import DMCTask, preprocess_dmc_data, get_stats, obtain_meta_data


_logger = logging.getLogger(__name__)

TRAIN_FILES = set()


TEST_FILES = set()

TRAIN_LENGTHS = {
    1: [14391, 2690, 3591, 3591, 2728, 14391],
    2: [28725, 4269, 35923],
    3: [28790, 28790, 28789, 28791],
    4: [7191, 28789, 28769, 28790, 28790, 86391],
    5: [28790, 28790, 28791, 4742, 3590, 28790, 7191, 2727],
    6: [28757, 28789, 28790, 28790, 3588, 86390, 28790, 53990, 7190, 2634, 2690, 2689],
    9: [28790, 3345, 86341, 14390, 86391, 53990],
    10: [14356, 13224, 28790, 28746, 28790, 28790, 35989]
}


TEST_LENGTHS = {
    1: [2945, 43233, 3632],
    2: [46791, 2883, 43230, 3631],
    3: [2482, 2620, 4231, 5937],
    4: [129591, 3632],
    5: [43191, 46791, 46810, 2489, 4232, 43230, 3629],
    6: [46807, 46785, 3629],
    9: [7506, 46808, 43259, 5938],
    10: [10284, 46807, 43230, 5930],
}

class DMCDataset(BaseTSDataset):
    """
    To-Do
    """
    GITHUB_LINK = 'https://github.com/exathlonbenchmark/exathlon.git'

    task_values = [e.value for e in DMCTask]

    def __init__(
            self,
            dataset_path: str = os.path.join(DATA_DIRECTORY, 'dmc'),
            app_id: Union[int, List[int]] = 1,
            training: bool = True,
            standardize: Union[bool,Callable[[pd.DataFrame, Dict],pd.DataFrame]] = True,
            use_normal_only: bool = True,
            preprocess: bool = True
            ):
        """

        :param dataset_path: Folder from which to load the dataset.
        :param app_id: Data from which app to load. Must be in [1-6, 9, 10].
        :param training: Whether to load the training or the test set.
        :param standardize: Can be either a bool that decides whether to apply the dataset-dependent default
            standardization or a function with signature (dataframe, stats) -> dataframe, where stats is a dictionary of
            common statistics on the training dataset (i.e., mean, std, median, etc. for each feature)
        :param download: Whether to download the dataset if it doesn't exist.
        :param preprocess: Whether to setup the dataset for experiments.
        """

        if isinstance(app_id, int):
            app_id = [app_id]

        self.app_id = []
        for i in app_id:
            if i not in self.task_values:
                raise ValueError(f'DMC Task must be one of {self.task_values}')
            else:
                self.app_id.append(DMCTask(i))

        
        
        self.dataset_path = dataset_path
        self.data_path = os.path.join(dataset_path, 'processed')
        self.train_data_path = os.path.join(self.data_path, 'train')
        self.test_data_path = os.path.join(self.data_path, 'test')

        self.training = training
        self.use_normal_only = use_normal_only

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it.')
        
        missing_tasks = self._search_missing_preprocessed_tasks()
        if len(missing_tasks) > 0:
            if not preprocess:
                raise RuntimeError('Dataset needs to be processed for proper working. Pass preprocess=True to setup the'
                                   ' dataset.')

            _logger.info("Processed data files not found! Running pre-processing now. This might take several minutes.")
            preprocess_dmc_data(
                missing_tasks=missing_tasks,
                out_data_dir=self.data_path,
                raw_data_dir=dataset_path
            )

        self.inputs = None
        self.targets = None

        self.intialize_meta_data(standardize=standardize)

    def intialize_meta_data(self, standardize):
    
        self.train_files, self.train_lengths, self.test_files, self.test_lengths = obtain_meta_data(
            path=self.data_path,
            tasks=self.app_id,
            use_normaly_only=self.use_normal_only,
        )

        stats = get_stats(
            path=self.train_data_path,
            tasks=self.app_id,
            use_normaly_only=self.use_normal_only
        )
        if callable(standardize):
            self.standardize_fn = functools.partial(standardize, stats=stats)
        elif standardize:
            self.standardize_fn = functools.partial(minmax_scaler, stats=stats)
        else:
            self.standardize_fn = None
        

    def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        load_path = self.train_data_path if self.training else self.test_data_path

        files = self.train_files if self.training else self.test_files

        inputs, targets = [], []
        for f in files:
            file_name = os.path.join(load_path, f)

            data = pd.read_csv(file_name, index_col='t')

            if self.training:
                target = np.zeros(len(data), dtype=np.int64)
            else:

                target = data['Anomaly'].to_numpy()
                target = target != 0
                target = target.astype(np.int64)

            data = data.drop(columns=['Anomaly'])

            if self.standardize_fn is not None:
                data = self.standardize_fn(data)
            data = data.astype(np.float32)

            input = data.to_numpy()

            inputs.append(input)
            targets.append(target)

        return inputs, targets

    def __getitem__(self, item: int) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        if not (0 <= item < len(self)):
            raise KeyError('Out of bounds')

        if self.inputs is None or self.targets is None:
            self.inputs, self.targets = self.load_data()

        return (torch.as_tensor(self.inputs[item]),), (torch.as_tensor(self.targets[item]),)

    def __len__(self) -> Optional[int]:
        return len(self.train_files) if self.training else len(self.test_files)

    @property
    def seq_len(self) -> List[int]:
        if self.training:
            return self.train_lengths
        else:
            return self.test_lengths

    @property
    def num_features(self) -> int:
        return 768

    @staticmethod
    def get_default_pipeline() -> Dict[str, Dict[str, Any]]:
        return {
            'subsample': {'class': 'SubsampleTransform', 'args': {'subsampling_factor': 5, 'aggregation': 'last'}},
            'cache': {'class': 'CacheTransform', 'args': {}}
        }

    @staticmethod
    def get_feature_names():
        column_names = [f"feature_{i}" for i in range(768)]
        return column_names

    def _check_exists(self) -> bool:
        # Only checks if the `data` folder exists
        data_types = ['test', 'train']
        features = ["normal_features", "random_features"]

        for data_type in data_types:
            for feature in features:
                for task in self.app_id:
                    data_folder_path = os.path.join(
                        self.dataset_path,
                        data_type,
                        feature,
                        task.name.lower()
                        )
                    if not os.path.isdir(data_folder_path):
                        return False
        return True

    def _search_missing_preprocessed_tasks(self) -> List[DMCTask]:
        # Only checks if the `processed` folder exsits
        data_types = ['test', 'train']
       

        missing_tasks = []
        for data_type in data_types:
            for task in self.app_id:
                data_folder_path = os.path.join(
                    self.data_path,
                    data_type,
                    task.name.lower()
                    )
                if not os.path.isdir(data_folder_path) and task not in missing_tasks:
                    missing_tasks.append(task)
        return missing_tasks