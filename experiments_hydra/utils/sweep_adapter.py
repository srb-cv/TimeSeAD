import copy
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import yaml

from timesead.utils.utils import param_grid_to_list_of_dicts


EXPERIMENT_MODULE_MAP = {
    "reconstruction.train_dense_ae": "experiments_hydra.reconstruction.train_dense_ae",
    "prediction.train_lstm_prediction_filonov": "experiments_hydra.prediction.train_lstm_prediction_filonov",
    "generative.vae.train_donut": "experiments_hydra.generative.vae.train_donut",
    "baselines.train_iqr_ad": "experiments_hydra.baselines.train_iqr_ad",
}

SECTION_NAME_MAP = {
    "model_params": "model",
    "detector_params": "detector",
}


def load_sweep_spec(path: str | Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_experiment_module(training_experiment: str) -> str:
    try:
        return EXPERIMENT_MODULE_MAP[training_experiment]
    except KeyError as exc:
        raise KeyError(
            f"No Hydra pilot mapping exists for training_experiment={training_experiment!r}. "
            f"Supported pilots: {sorted(EXPERIMENT_MODULE_MAP)}"
        ) from exc


def _recursive_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _recursive_update(base[key], value)
        else:
            base[key] = value
    return base


def _remap_sections(data: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in data.items():
        key = SECTION_NAME_MAP.get(key, key)
        if isinstance(value, dict):
            result[key] = _remap_sections(value)
        else:
            result[key] = value
    return result


def _encode_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    if isinstance(value, str):
        if any(ch in value for ch in (" ", ",", "=", ":", "[", "]", "{", "}")):
            return json.dumps(value)
        return value

    return json.dumps(value)


def _flatten_overrides(obj: Dict[str, Any], prefix: str = "") -> Iterator[Tuple[str, Any]]:
    for key, value in obj.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from _flatten_overrides(value, prefix=full_key)
        else:
            yield full_key, value


def build_hydra_overrides(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    updates = _remap_sections(copy.deepcopy(spec.get("training_param_updates", {})))
    training_grid = _remap_sections(copy.deepcopy(spec.get("training_param_grid", {})))
    detector_grid = _remap_sections(copy.deepcopy(spec.get("detector_param_grid", {})))

    training_points = list(param_grid_to_list_of_dicts(training_grid)) or [{}]
    detector_points = list(param_grid_to_list_of_dicts(detector_grid)) or [{}]

    points: List[Dict[str, Any]] = []
    for train_point, detector_point in itertools.product(training_points, detector_points):
        merged = copy.deepcopy(updates)
        _recursive_update(merged, train_point)
        _recursive_update(merged, detector_point)
        points.append(merged)

    return points


def format_hydra_override_strings(point: Dict[str, Any], extra_overrides: Iterable[str] = ()) -> List[str]:
    # This function converts a config dictionary into Hydra command-line override strings.
    #     point = {
    #   "model.lr": 0.001,
    #   "trainer.batch_size": 32 }
    # => [
    #   "++model.lr=0.001",
    #   "++trainer.batch_size=32"
    # ]
    overrides = []
    for key, value in _flatten_overrides(point):
        encoded = _encode_override_value(value)
        overrides.append(f"++{key}={encoded}")

    overrides.extend(extra_overrides)
    return overrides
