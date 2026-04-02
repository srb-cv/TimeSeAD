import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import mlflow
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from timesead.evaluation import Evaluator
from timesead.models.baselines import IQRAnomalyDetector
from timesead.models.common import MSEReconstructionAnomalyDetector
from timesead.models.generative import DonutAnomalyDetector
from timesead.models.prediction import LSTMPredictionAnomalyDetector, LSTMS2SPredictionAnomalyDetector
from timesead.utils.metadata import PROJECT_ROOT

from experiments_hydra.utils import get_dataloader, load_dataset
from experiments_hydra.utils.config import to_plain_config


def _resolve_tracking_uri(training_cfg: Dict[str, Any], override: Optional[str]) -> str:
    if override is not None:
        return override

    tracking_uri = training_cfg["experiment"].get("tracking_uri")
    if tracking_uri is not None:
        return tracking_uri

    return f"file://{PROJECT_ROOT}/mlruns_hydra"


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _sanitize_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _get_subseq_lengths_and_window_size(dataset) -> Tuple[List[int], int]:
    transform = dataset.sink_transform
    window_size = getattr(transform, "input_window_size", None)
    if window_size is None:
        window_size = getattr(transform, "window_size", None)

    parent = getattr(transform, "parent", None)
    subseq_lengths = parent.seq_len if parent is not None else dataset.seq_len
    if isinstance(subseq_lengths, int):
        subseq_lengths = [subseq_lengths] * len(parent if parent is not None else dataset)

    return subseq_lengths, window_size


def _build_eval_dataset_cfg(training_cfg: Dict[str, Any], cli_cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    dataset_cfg = copy.deepcopy(training_cfg["dataset"])
    dataset_cfg.setdefault("ds_args", {})
    dataset_cfg["ds_args"]["training"] = False
    test_pipeline = dataset_cfg.get("test_pipeline")
    if test_pipeline is None:
        raise ValueError(
            "Hydra evaluation requires dataset.test_pipeline to be defined explicitly in the experiment config."
        )
    dataset_cfg["pipeline"] = copy.deepcopy(test_pipeline)

    split = cli_cfg.get("split")
    if split is None:
        split = [0.3, 0.7]
    dataset_cfg["split"] = split

    split_axis = cli_cfg.get("split_axis")
    if split_axis is None:
        split_axis = training_cfg["dataset"].get("split_axis", "time")
    dataset_cfg["split_axis"] = split_axis

    split_index = cli_cfg.get("split_index")
    if split_index is None:
        split_index = 1 if len(split) > 1 else 0

    return dataset_cfg, split_index


def _load_detector(run_dir: Path, training_cfg: Dict[str, Any], device: str):
    artifact_path = run_dir / "artifacts" / "final_model.pth"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Expected artifact {artifact_path} does not exist.")

    payload = torch.load(artifact_path, map_location=device, weights_only=False)
    detector = payload.get("detector")
    model = payload.get("model")

    if detector is None:
        family = training_cfg["experiment"]["family"]
        if family == "reconstruction":
            detector = MSEReconstructionAnomalyDetector(model, batch_first=True)
        elif family == "prediction":
            window_size = training_cfg["dataset"]["pipeline"]["prediction"]["args"]["window_size"]
            detector = LSTMS2SPredictionAnomalyDetector(model, half_life=2 * window_size)
        elif family == "generative":
            detector = DonutAnomalyDetector(model, num_mc_samples=training_cfg["detector"]["num_mc_samples"])
        elif family == "baseline":
            raise RuntimeError("Baseline evaluation requires a saved fitted detector, but final_model.pth has none.")
        else:
            raise RuntimeError(f"Unsupported family for detector reconstruction: {family}")

    return detector.to(device)


def _compute_metrics(training_cfg: Dict[str, Any], cli_cfg: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    eval_dataset_cfg, split_index = _build_eval_dataset_cfg(training_cfg, cli_cfg) # split_index = 1

    # Load test dataset
    datasets = load_dataset(**eval_dataset_cfg)
    eval_ds = datasets[split_index]
    loader = get_dataloader(eval_ds, {**training_cfg["training"], "shuffle": False})

    ########## Computing AD Score ##########
    detector = _load_detector(run_dir, training_cfg, training_cfg["training"]["device"])
    detector.eval()

    if isinstance(detector, LSTMPredictionAnomalyDetector):
        subseq_lengths, window_size = _get_subseq_lengths_and_window_size(loader.dataset)
        labels, scores = detector.get_labels_and_scores(loader, subseq_lengths=subseq_lengths, window_size=window_size)
    else:
        labels, scores = detector.get_labels_and_scores(loader)
    ##############################################


    requested_metrics = cli_cfg.get("metric_names")
    if requested_metrics is None:
        requested_metrics = [training_cfg["sweep"]["validation_metric"], *training_cfg["sweep"].get("evaluation_metrics", [])]

    metric_names: List[str] = []
    for metric in requested_metrics:
        if metric not in metric_names:
            metric_names.append(metric)

    ########## Computing Metrics ##########
    evaluator = Evaluator()
    metrics = {}
    validation_metric = training_cfg["sweep"]["validation_metric"]
    for metric_name in metric_names:
        try:
            score, info = getattr(evaluator, metric_name)(labels, scores)
            metrics[metric_name] = {"score": score, "info": info}
        except Exception as exc:
            if metric_name == validation_metric:
                raise
            metrics[metric_name] = {"score": None, "info": {"error": str(exc)}}
    ##############################################

    return {
        "metrics": metrics,
        "num_scores": int(scores.shape[0]),
        "dataset": eval_dataset_cfg,
        "split_index": split_index,
    }


def _maybe_log_to_mlflow(training_cfg: Dict[str, Any], cli_cfg: Dict[str, Any], summary_path: Path, summary: Dict[str, Any]) -> None:
    if not cli_cfg.get("log_to_mlflow", True):
        return

    run_id = cli_cfg.get("run_id")
    if run_id is None:
        run_id_file = Path(cli_cfg["run_dir"]) / "mlflow_run_id.txt"
        if run_id_file.exists():
            run_id = run_id_file.read_text(encoding="utf-8").strip() or None

    if run_id is None:
        return

    mlflow.set_tracking_uri(_resolve_tracking_uri(training_cfg, cli_cfg.get("tracking_uri")))
    with mlflow.start_run(run_id=run_id):
        for metric_name, metric_payload in summary["metrics"].items():
            if metric_payload["score"] is None:
                continue
            mlflow.log_metric(metric_name, float(metric_payload["score"]))
        mlflow.log_artifact(str(summary_path), artifact_path="artifacts")


@hydra.main(version_base=None, config_path="configs/evaluate", config_name="default")
def run(cfg) -> Dict[str, Any]:

    ########## 1. Configuration Loading ##########
    cfg = to_plain_config(cfg)
    if cfg["run_dir"] is None:
        raise ValueError("Set run_dir to the Hydra output directory of a finished training run.")

    run_dir = Path(cfg["run_dir"]).resolve()
    hydra_config_path = run_dir / ".hydra" / "config.yaml"
    if not hydra_config_path.exists():
        raise FileNotFoundError(f"Hydra config file {hydra_config_path} does not exist.")

    training_cfg = to_plain_config(OmegaConf.load(hydra_config_path))

    ##############################################
    # cfg: the configuration of evaluate.py
    # training_cfg: the configuration of train
    # run_dir: the output direction of model
    summary = _compute_metrics(training_cfg, cfg, run_dir)

    summary = _sanitize_json(summary)

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / cfg["artifact_name"]
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    _maybe_log_to_mlflow(training_cfg, cfg, summary_path, summary)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
