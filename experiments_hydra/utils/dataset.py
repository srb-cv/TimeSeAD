import collections.abc
import copy
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Sampler

from timesead.data.dataset import collate_fn
from timesead.data.transforms import PipelineDataset, make_dataset_split, make_pipe_from_dict
from timesead.utils.utils import objspec2constructor

from .config import to_plain_config


def _recursive_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _recursive_update(base[key], value)
        else:
            base[key] = value

    return base


def load_dataset(
    name: str,
    ds_args: Dict[str, Any],
    pipeline: Dict[str, Any],
    use_dataset_pipeline: bool,
    split: Tuple[float, ...],
    split_axis: str,
    test_pipeline: Dict[str, Any] = None,
) -> List[PipelineDataset]:
    ds = objspec2constructor({"class": name, "args": ds_args}, base_module="timesead.data")()

    if not isinstance(pipeline, collections.abc.Sequence) or isinstance(pipeline, dict):
        pipelines = [pipeline] * len(split)
    else:
        pipelines = pipeline

    assert len(pipelines) == len(split)

    ds_splits = make_dataset_split(ds, *split, axis=split_axis)
    result = []
    for split_pipeline, ds_pipe in zip(pipelines, ds_splits):
        split_pipeline = copy.deepcopy(split_pipeline)
        if use_dataset_pipeline:
            default_pipe = copy.deepcopy(ds.get_default_pipeline())
            default_pipe = _recursive_update(default_pipe, split_pipeline)
            split_pipeline = default_pipe

        pipe = make_pipe_from_dict(split_pipeline, ds_pipe)
        result.append(pipe)

    return result


def get_dataloader(dataset, training_cfg, sampler:Sampler=None):
    training_cfg = to_plain_config(training_cfg)
    dataloader = None

    if sampler != None:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,   # 👈 replace batch_size + shuffle
            num_workers=training_cfg["num_workers"],
            collate_fn=collate_fn(training_cfg["batch_dim"]),
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=training_cfg["batch_size"],
            num_workers=training_cfg["num_workers"],
            shuffle=training_cfg.get("shuffle", True),
            collate_fn=collate_fn(training_cfg["batch_dim"]),
            drop_last=training_cfg["drop_last"],
        )
        
    return dataloader

def get_data_labels(dataset: PipelineDataset):
    labels = []
    for _, targets in dataset:
        label = targets[0].max().item()
        labels.append(label)

    return labels