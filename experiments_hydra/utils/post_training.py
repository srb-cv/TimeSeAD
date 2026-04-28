from pathlib import Path
from typing import Any, Optional

from experiments_hydra.evaluate import evaluate_run, evaluate_run_with_holdout

from .config import to_plain_config


def should_evaluate_after_training(cfg: Any) -> bool:
    plain_cfg = to_plain_config(cfg)
    experiment_cfg = plain_cfg.get("experiment", {})
    return bool(experiment_cfg.get("evaluate_after_training", False))


def maybe_evaluate_after_training(
    cfg: Any, output_dir: str | Path
) -> Optional[dict]:
    if not should_evaluate_after_training(cfg):
        return None

    plain_cfg = to_plain_config(cfg)
    experiment_cfg = plain_cfg.get("experiment", {})
    post_train_cfg = dict(experiment_cfg.get("post_train_evaluation", {}))
    protocol = post_train_cfg.pop("protocol", "single_split")

    if protocol == "single_split":
        return evaluate_run(run_dir=output_dir, **post_train_cfg)
    if protocol == "holdout_calibration":
        print(
            "Evaluating with holdout calibration protocol."
        )
        return evaluate_run_with_holdout(run_dir=output_dir, **post_train_cfg)

    raise ValueError(
        f"Unsupported post-train evaluation protocol {protocol!r}."
    )
