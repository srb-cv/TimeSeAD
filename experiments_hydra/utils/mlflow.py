import contextlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import mlflow

from timesead.utils.metadata import PROJECT_ROOT

from .config import flatten_config, to_plain_config


class MLflowMetricLogger:
    def __init__(self):
        self.metric_steps = defaultdict(int)

    def log_metric(self, metric_name: str, metric_value: Any) -> None:
        try:
            value = float(metric_value)
        except (TypeError, ValueError):
            return

        step = self.metric_steps[metric_name]
        mlflow.log_metric(metric_name, value, step=step)
        self.metric_steps[metric_name] += 1


def log_hydra_run_reference(output_dir: str | Path) -> Optional[Dict[str, str]]:
    active_run = mlflow.active_run()
    if active_run is None:
        return None

    resolved_output_dir = Path(output_dir).resolve()
    tags = {
        "hydra_run_dir": str(resolved_output_dir),
        "hydra_artifact_dir": str(resolved_output_dir / "artifacts"),
        "hydra_config_path": str(resolved_output_dir / ".hydra" / "config.yaml"),
    }
    mlflow.set_tags(tags)
    return tags


def save_active_run_id(output_dir: str) -> Optional[Path]:
    active_run = mlflow.active_run()
    if active_run is None:
        return None

    resolved_output_dir = Path(output_dir).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = resolved_output_dir / "mlflow_run_id.txt"
    output_path.write_text(active_run.info.run_id, encoding="utf-8")
    log_hydra_run_reference(resolved_output_dir)
    return output_path


@contextlib.contextmanager
def start_mlflow_run(
    cfg, run_name: Optional[str] = None, output_dir: Optional[str | Path] = None
) -> Iterator[MLflowMetricLogger]:
    cfg = to_plain_config(cfg)
    tracking_uri = cfg["experiment"].get("tracking_uri")
    if tracking_uri is None:
        tracking_uri = f"sqlite:///{PROJECT_ROOT}/mlruns_hydra/mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = cfg["experiment"].get("mlflow_experiment_name", cfg["experiment"]["name"])
    mlflow.set_experiment(experiment_name)

    tags: Dict[str, str] = {
        "framework": "hydra+mlflow",
        "timesead_family": cfg["experiment"].get("family", "unknown"),
    }
    for key, value in cfg["experiment"].get("tags", {}).items():
        tags[key] = str(value)

    params = flatten_config(
        {
            "dataset": cfg.get("dataset", {}),
            "training": cfg.get("training", {}),
            "model": cfg.get("model", {}),
            "detector": cfg.get("detector", {}),
        }
    )

    with mlflow.start_run(run_name=run_name or cfg["experiment"]["name"], tags=tags):
        if params:
            mlflow.log_params(params)
        if output_dir is not None:
            log_hydra_run_reference(output_dir)
        yield MLflowMetricLogger()
