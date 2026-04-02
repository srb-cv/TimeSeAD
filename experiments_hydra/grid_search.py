import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import mlflow
import yaml
from hydra.core.hydra_config import HydraConfig
from mlflow.tracking import MlflowClient

from timesead.utils.metadata import PROJECT_ROOT

from experiments_hydra.utils.sweep_adapter import (
    EXPERIMENT_MODULE_MAP,
    build_hydra_overrides,
    format_hydra_override_strings,
    get_experiment_module,
    load_sweep_spec,
)


def _select_better(candidate: float, best: Optional[float], mode: str) -> bool:
    if best is None:
        return True
    if mode == "max":
        return candidate > best
    return candidate < best


def _resolve_tracking_uri(config_tracking_uri: Optional[str]) -> str:
    if config_tracking_uri:
        return config_tracking_uri
    return f"file://{PROJECT_ROOT}/mlruns_hydra"


def _load_hydra_experiment_spec(config_name: str) -> Dict[str, Any]:
    config_file = Path(PROJECT_ROOT) / "experiments_hydra" / "configs" / f"{config_name}.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"Hydra experiment config {config_file} does not exist.")

    with open(config_file, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_run_metric(client: MlflowClient, experiment_name: str, sweep_id: str, grid_point: int, metric_name: str):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment {experiment_name!r} was not created.")

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.sweep_id = '{sweep_id}' and tags.grid_point = '{grid_point}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No MLflow run found for sweep_id={sweep_id} grid_point={grid_point}.")

    run = runs[0]
    metric = run.data.metrics.get(metric_name)
    return run, metric


def _load_run(client: MlflowClient, experiment_name: str, sweep_id: str, grid_point: int):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment {experiment_name!r} was not created.")

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.sweep_id = '{sweep_id}' and tags.grid_point = '{grid_point}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No MLflow run found for sweep_id={sweep_id} grid_point={grid_point}.")

    return runs[0]


@hydra.main(version_base=None, config_path="configs/grid_search", config_name="default")
def run(cfg) -> Dict[str, Any]:
    if cfg.sweep.experiment_config is not None:
        spec = _load_hydra_experiment_spec(cfg.sweep.experiment_config)
        module_name = spec["experiment"]["entrypoint"]
        points = build_hydra_overrides(
            {
                "training_param_updates": spec.get("sweep", {}).get("training_param_updates", {}),
                "training_param_grid": spec.get("sweep", {}).get("training_param_grid", {}),
                "detector_param_grid": spec.get("sweep", {}).get("detector_param_grid", {}),
            }
        )
        selection_metric = (
            cfg.sweep.selection_metric
            or spec.get("sweep", {}).get("selection_metric")
            or spec.get("sweep", {}).get("validation_metric")
        )
        mode = cfg.sweep.mode or spec.get("sweep", {}).get("mode", "min")
        config_name_override = cfg.sweep.experiment_config
    else:
        if cfg.sweep.config_path is None:
            raise ValueError(
                "Set either sweep.experiment_config to a new Hydra config name or "
                "sweep.config_path to an existing experiment_configs/*.yml file."
            )

        spec = load_sweep_spec(cfg.sweep.config_path)
        params = spec["params"]
        training_experiment = params["training_experiment"]
        module_name = get_experiment_module(training_experiment)

        # building different hyperparameters for training + detector
        points = build_hydra_overrides(spec)
        selection_metric = cfg.sweep.selection_metric or params.get("selection_metric") or params["validation_metric"]
        mode = cfg.sweep.mode
        config_name_override = None

    if cfg.sweep.max_runs is not None:
        points = points[: cfg.sweep.max_runs]

    tracking_uri = _resolve_tracking_uri(cfg.sweep.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri) # file:///home/cor54gyp/TimeSeAD/timesead/utils/../..//mlruns_hydra
    client = MlflowClient(tracking_uri=tracking_uri)

    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
    sweep_output_dir = Path(cfg.sweep.output_dir) if cfg.sweep.output_dir is not None else hydra_output_dir
    sweep_id = hydra_output_dir.name
    run_summaries: List[Dict[str, Any]] = []
    best_summary: Optional[Dict[str, Any]] = None

    for idx, point in enumerate(points):
        run_output_dir = sweep_output_dir / f"grid_{idx:03d}"
        extra = [
            f"+experiment.tags.sweep_id={sweep_id}",
            f"+experiment.tags.grid_point={idx}",
            f"+experiment.tags.sweep_source={cfg.sweep.experiment_config or Path(cfg.sweep.config_path).name}",
            f"hydra.run.dir={run_output_dir}",
        ]
        if cfg.sweep.tracking_uri is not None:
            extra.append(f"experiment.tracking_uri={cfg.sweep.tracking_uri}")
        extra.extend(cfg.sweep.extra_overrides)

        overrides = format_hydra_override_strings(point, extra_overrides=extra)
        cmd = [sys.executable, "-m", module_name]
        if config_name_override is not None:
            cmd.extend(["--config-name", config_name_override])
        cmd.extend(overrides)
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

        run = _load_run(client, experiment_name="TimeSeAD-Hydra", sweep_id=sweep_id, grid_point=idx)

        if cfg.sweep.run_evaluation:
            eval_cmd = [
                sys.executable,
                "-m",
                "experiments_hydra.evaluate",
                f"run_dir={run_output_dir}",
                f"run_id={run.info.run_id}",
            ]
            if cfg.sweep.tracking_uri is not None:
                eval_cmd.append(f"tracking_uri={cfg.sweep.tracking_uri}")
            subprocess.run(eval_cmd, cwd=PROJECT_ROOT, check=True)

        run, metric = _load_run_metric(
            client,
            experiment_name="TimeSeAD-Hydra",
            sweep_id=sweep_id,
            grid_point=idx,
            metric_name=selection_metric,
        )
        if metric is None and cfg.sweep.fail_on_missing_metric:
            raise RuntimeError(
                f"Run {run.info.run_id} did not log selection metric {selection_metric!r}. "
                f"Either set sweep.selection_metric to a logged metric such as val_loss_0 or disable fail_on_missing_metric."
            )

        summary = {
            "grid_point": idx,
            "run_id": run.info.run_id,
            "metric_name": selection_metric,
            "metric_value": metric,
            "overrides": overrides,
            "artifact_uri": run.info.artifact_uri,
        }
        run_summaries.append(summary)

        if metric is not None and _select_better(metric, None if best_summary is None else best_summary["metric_value"], mode):
            best_summary = summary

    hydra_output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = hydra_output_dir / "sweep_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({"best_run": best_summary, "runs": run_summaries}, handle, indent=2)

    if best_summary is not None:
        print(json.dumps(best_summary, indent=2))
    else:
        print(json.dumps({"message": "No best run selected.", "runs": run_summaries}, indent=2))

    return {"best_run": best_summary, "runs": run_summaries}


if __name__ == "__main__":
    run()
