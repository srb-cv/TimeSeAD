import json
from typing import Any, Dict

from omegaconf import DictConfig, ListConfig, OmegaConf


def to_plain_config(cfg: Any) -> Any:
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.to_container(cfg, resolve=True)

    return cfg


def flatten_config(cfg: Any, prefix: str = "") -> Dict[str, str]:
    cfg = to_plain_config(cfg)

    if isinstance(cfg, dict):
        result: Dict[str, str] = {}
        for key, value in cfg.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_config(value, prefix=child_prefix))
        return result

    if isinstance(cfg, list):
        return {prefix: json.dumps(cfg)}

    if cfg is None:
        return {prefix: "null"}

    return {prefix: str(cfg)}

