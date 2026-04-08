import functools
import os
from typing import Tuple, Optional, Union, Callable, Dict, Any, List
import logging
import json

import numpy as np
import pandas as pd
import torch

from timesead.data.dataset import BaseTSDataset
from timesead.data.preprocessing import minmax_scaler
from timesead.utils.metadata import DATA_DIRECTORY
from timesead.data.preprocessing.dmc import DMCTask, construct_meta_data, get_stats, obtain_meta_data, META_DATASET_FILE, parse_meta_data


_logger = logging.getLogger(__name__)



class DMCDataset(BaseTSDataset):
    """
    To-Do
    """
    GITHUB_LINK = 'https://github.com/exathlonbenchmark/exathlon.git'

    task_values = [e.value for e in DMCTask]

    def __init__(
            self,
            dataset_path: str = os.path.join(DATA_DIRECTORY, 'dmc'),
            task_id: Union[int, List[int]] = 1, # change name
            training: bool = True,
            standardize: Union[bool,Callable[[pd.DataFrame, Dict],pd.DataFrame]] = True,
            use_normal_only: bool = True,
            preprocess: bool = True
            ):
        """

        :param dataset_path: Folder from which to load the dataset.
        :param task_id: Data from which app to load. Must be in [1-6, 9, 10].
        :param training: Whether to load the training or the test set.
        :param standardize: Can be either a bool that decides whether to apply the dataset-dependent default
            standardization or a function with signature (dataframe, stats) -> dataframe, where stats is a dictionary of
            common statistics on the training dataset (i.e., mean, std, median, etc. for each feature)
        :param download: Whether to download the dataset if it doesn't exist.
        :param preprocess: Whether to setup the dataset for experiments.
        """

        if isinstance(task_id, int):
            task_id = [task_id]

        self.task_id = []
        for i in task_id:
            if i not in self.task_values:
                raise ValueError(f'DMC Task must be one of {self.task_values}')
            else:
                self.task_id.append(DMCTask(i))

        
        
        self.dataset_path = dataset_path
        self.train_dataset_path = os.path.join(self.dataset_path, 'train')
        self.test_dataset_path = os.path.join(self.dataset_path, 'test')

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
        construct_meta_data(
            missing_tasks = missing_tasks,
            data_dir = self.dataset_path)

        self.inputs = None
        self.targets = None

        self.intialize_meta_data(standardize=standardize)

    def intialize_meta_data(self, standardize):
        json_file = os.path.join(self.dataset_path, META_DATASET_FILE)
        self.train_meta_datas, self.test_meta_datas = obtain_meta_data(
            json_path=json_file,
            tasks=self.task_id,
            use_normaly_only=self.use_normal_only,
        )

        self.train_files, self.train_lengths, self.train_labels = parse_meta_data(self.train_meta_datas)
        self.test_files, self.test_lengths, self.test_labels = parse_meta_data(self.test_meta_datas)

        stats = get_stats(
            path=self.train_dataset_path,
            tasks=self.task_id,
            use_normaly_only=self.use_normal_only
        )
        if callable(standardize):
            self.standardize_fn = functools.partial(standardize, stats=stats)
        elif standardize:
            self.standardize_fn = functools.partial(minmax_scaler, stats=stats)
        else:
            self.standardize_fn = None
        

    def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        load_path = self.train_dataset_path if self.training else self.test_dataset_path

        files = self.train_meta_datas if self.training else self.test_meta_datas

        inputs, targets = [], []
        for f in files:
            file_name = f[0]
            file_label = f[-1]
            file_name = os.path.join(load_path, file_name)

            data = np.load(file_name)['features']
    
            if file_label:
                target = np.ones(data.shape[0])
            else:
                target = np.zeros(data.shape[0])
            
            target = target.astype(np.int64)

            if self.standardize_fn is not None:
                data = self.standardize_fn(data)
            data = data.astype(np.float32)

            input = data #.to_numpy()

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
        return len(self.train_meta_datas) if self.training else len(self.test_meta_datas)

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
                for task in self.task_id:
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
        json_file = os.path.join(self.dataset_path, META_DATASET_FILE)

        missing_tasks = []
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                meta_dict = json.load(f)
            
            for task in self.task_id:
                if not task.name.lower() in meta_dict:
                    missing_tasks.append(task)
        else:
            missing_tasks = self.task_id
            
        return missing_tasks