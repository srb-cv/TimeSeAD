import copy
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import hydra
import mlflow
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from sklearn.metrics import f1_score as sklearn_f1_score

from timesead.evaluation import Evaluator
from timesead.evaluation.ts_precision_recall import (
    improved_cardinality_fn,
    inverse_proportional_cardinality_fn,
    ts_precision_and_recall,
)
from timesead.models.baselines import IQRAnomalyDetector
from timesead.models.common import MSEReconstructionAnomalyDetector
from timesead.models.generative import DonutAnomalyDetector
from timesead.models.prediction import (
    LSTMPredictionAnomalyDetector,
    LSTMS2SPredictionAnomalyDetector,
)
from timesead.utils.metadata import PROJECT_ROOT

from experiments_hydra.utils import get_dataloader, load_dataset
from experiments_hydra.utils.config import to_plain_config
from experiments_hydra.utils.mlflow import log_hydra_run_reference


THRESHOLD_CALIBRATED_METRIC_NAMES = {
    "best_f1_score": "f1_score",
    "best_ts_f1_score": "ts_f1_score",
    "best_ts_f1_score_classic": "ts_f1_score_classic",
}


def _recursive_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _recursive_update(base[key], value)
        else:
            base[key] = value

    return base


def _resolve_tracking_uri(training_cfg: Dict[str, Any], override: Optional[str]) -> str:
    if override is not None:
        return override

    tracking_uri = training_cfg["experiment"].get("tracking_uri")
    if tracking_uri is not None:
        return tracking_uri

    return f"sqlite:///{PROJECT_ROOT}/mlruns_hydra/mlflow.db"


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


def load_training_cfg(run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    hydra_config_path = run_dir / ".hydra" / "config.yaml"
    if not hydra_config_path.exists():
        raise FileNotFoundError(
            f"Hydra config file {hydra_config_path} does not exist."
        )

    return to_plain_config(OmegaConf.load(hydra_config_path))


def compute_val_splits(
    folds: int, val_fold: int, padding: int = 1
) -> Tuple[List[int], int, List[int]]:
    if folds <= 0:
        raise ValueError("folds must be positive.")
    if not 0 <= val_fold < folds:
        raise ValueError(f"val_fold must be in [0, {folds}), got {val_fold}.")

    if folds == 1:
        return [1], 0, []

    splits = [
        val_fold - padding,
        min(val_fold, padding),
        1,
        min(folds - val_fold - 1, padding),
        folds - val_fold - 1 - padding,
    ]

    val_fold_index = 2
    if splits[0] <= 0:
        val_fold_index -= 1
    if splits[1] <= 0:
        val_fold_index -= 1

    splits = [split for split in splits if split > 0]

    test_fold_indices = list(range(len(splits)))
    test_fold_indices.remove(val_fold_index)
    if padding > 0:
        if val_fold_index > 0:
            test_fold_indices.remove(val_fold_index - 1)
        if val_fold_index < len(splits) - 1:
            test_fold_indices.remove(val_fold_index + 1)

    return splits, val_fold_index, test_fold_indices


def _get_subseq_lengths_and_window_size(dataset) -> Tuple[List[int], int]:
    transform = dataset.sink_transform
    window_size = getattr(transform, "input_window_size", None)
    if window_size is None:
        window_size = getattr(transform, "window_size", None)
    parent = getattr(transform, "parent", None)
    subseq_lengths = parent.seq_len if parent is not None else dataset.seq_len
    if isinstance(subseq_lengths, int):
        subseq_lengths = [subseq_lengths] * len(
            parent if parent is not None else dataset
        )

    return subseq_lengths, window_size


def build_training_dataset_cfg(
    training_cfg: Dict[str, Any],
    split: Optional[Sequence[float]] = None,
    split_axis: Optional[str] = None,
) -> Dict[str, Any]:
    dataset_cfg = copy.deepcopy(training_cfg["dataset"])
    dataset_cfg.setdefault("ds_args", {})
    dataset_cfg["ds_args"]["training"] = True
    dataset_cfg["pipeline"] = copy.deepcopy(training_cfg["dataset"]["pipeline"])

    if split is not None:
        dataset_cfg["split"] = list(split)
    if split_axis is not None:
        dataset_cfg["split_axis"] = split_axis

    return dataset_cfg


def build_labelled_test_dataset_cfg(
    training_cfg: Dict[str, Any],
    split: Optional[Sequence[float]] = None,
    split_axis: Optional[str] = None,
    pipeline_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    dataset_cfg = copy.deepcopy(training_cfg["dataset"])
    dataset_cfg.setdefault("ds_args", {})
    dataset_cfg["ds_args"]["training"] = False
    test_pipeline = dataset_cfg.get("test_pipeline")
    if test_pipeline is None:
        raise ValueError(
            "Hydra evaluation requires dataset.test_pipeline to be defined explicitly in the experiment config."
        )
    dataset_cfg["pipeline"] = copy.deepcopy(test_pipeline)
    if pipeline_overrides:
        _recursive_update(dataset_cfg["pipeline"], copy.deepcopy(pipeline_overrides))

    if split is None:
        split = [0.3, 0.7]
    dataset_cfg["split"] = list(split)

    if split_axis is None:
        split_axis = training_cfg["dataset"].get("split_axis", "time")
    dataset_cfg["split_axis"] = split_axis

    return dataset_cfg


def load_split_datasets(dataset_cfg: Dict[str, Any]):
    return load_dataset(**dataset_cfg)


def build_training_validation_loader(
    training_cfg: Dict[str, Any], split_index: Optional[int] = None
):
    dataset_cfg = build_training_dataset_cfg(training_cfg)
    datasets = load_split_datasets(dataset_cfg)
    resolved_split_index = _resolve_split_index(dataset_cfg["split"], split_index)
    validation_dataset = datasets[resolved_split_index]
    loader = get_dataloader(
        validation_dataset, {**training_cfg["training"], "shuffle": False}
    )
    return dataset_cfg, validation_dataset, loader


def build_labelled_test_splits(
    training_cfg: Dict[str, Any],
    split: Optional[Sequence[float]] = None,
    split_axis: Optional[str] = None,
    pipeline_overrides: Optional[Dict[str, Any]] = None,
):
    dataset_cfg = build_labelled_test_dataset_cfg(
        training_cfg,
        split=split,
        split_axis=split_axis,
        pipeline_overrides=pipeline_overrides,
    )
    return dataset_cfg, load_split_datasets(dataset_cfg)


def load_model_artifact(run_dir: str | Path, device: Optional[str] = None) -> Dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    artifact_path = run_dir / "artifacts" / "final_model.pth"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Expected artifact {artifact_path} does not exist.")

    return torch.load(artifact_path, map_location=device, weights_only=False)


def _resolve_detector_cfg(
    training_cfg: Dict[str, Any], detector_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    detector_cfg = copy.deepcopy(training_cfg.get("detector", {}))
    if detector_overrides is None:
        return detector_cfg

    overrides = copy.deepcopy(detector_overrides)
    if "detector" in overrides and isinstance(overrides["detector"], dict):
        overrides = overrides["detector"]

    return _recursive_update(detector_cfg, overrides)


def build_detector_from_artifact(
    artifact_payload: Dict[str, Any],
    training_cfg: Dict[str, Any],
    detector_overrides: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    prefer_saved_detector: bool = True,
):
    resolved_device = device or training_cfg["training"]["device"]
    saved_detector = artifact_payload.get("detector")
    model = artifact_payload.get("model")
    if (
        prefer_saved_detector
        and detector_overrides in (None, {})
        and saved_detector is not None
    ):
        return saved_detector.to(resolved_device)

    detector_cfg = _resolve_detector_cfg(training_cfg, detector_overrides)
    family = training_cfg["experiment"]["family"]
    if family == "reconstruction":
        detector = MSEReconstructionAnomalyDetector(
            model, batch_first=detector_cfg.get("batch_first", True)
        )
    elif family == "prediction":
        half_life = detector_cfg.get("half_life")
        if half_life is None:
            window_size = training_cfg["dataset"]["pipeline"]["prediction"]["args"][
                "window_size"
            ]
            half_life = 2 * window_size
        detector = LSTMS2SPredictionAnomalyDetector(model, half_life=half_life)
    elif family == "generative":
        detector = DonutAnomalyDetector(
            model,
            num_mc_samples=detector_cfg.get("num_mc_samples", 1024),
        )
    elif family == "baseline":
        detector = IQRAnomalyDetector(
            std_factor=detector_cfg.get("std_factor", 2.58),
            first_diffs=detector_cfg.get("first_diffs", False),
            cum_method=detector_cfg.get("cum_method", "mean"),
            feature_index=detector_cfg.get("feature_index"),
            input_shape=detector_cfg.get("input_shape", "btf"),
        )
    elif saved_detector is None:
        raise RuntimeError(f"Unsupported family for detector reconstruction: {family}")
    else:
        detector = saved_detector

    return detector.to(resolved_device)


def build_detector_from_run(
    run_dir: str | Path,
    detector_overrides: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    prefer_saved_detector: bool = True,
):
    training_cfg = load_training_cfg(run_dir)
    artifact_payload = load_model_artifact(
        run_dir, device=device or training_cfg["training"]["device"]
    )
    return build_detector_from_artifact(
        artifact_payload,
        training_cfg,
        detector_overrides=detector_overrides,
        device=device,
        prefer_saved_detector=prefer_saved_detector,
    )


def fit_detector_if_supported(
    detector,
    fit_loader,
    fit_kwargs: Optional[Dict[str, Any]] = None,
):
    if fit_loader is None:
        return detector

    resolved_fit_kwargs = dict(fit_kwargs or {})
    fit_signature = inspect.signature(detector.fit)
    if isinstance(detector, LSTMPredictionAnomalyDetector):
        accepts_subseq_lengths = "subseq_lengths" in fit_signature.parameters
        accepts_window_size = "window_size" in fit_signature.parameters
        if accepts_subseq_lengths and "subseq_lengths" not in resolved_fit_kwargs:
            subseq_lengths, window_size = _get_subseq_lengths_and_window_size(
                fit_loader.dataset
            )
            resolved_fit_kwargs["subseq_lengths"] = subseq_lengths
            if accepts_window_size and "window_size" not in resolved_fit_kwargs:
                resolved_fit_kwargs["window_size"] = window_size

    detector.fit(fit_loader, **resolved_fit_kwargs)
    return detector


def _resolve_metric_names(
    training_cfg: Dict[str, Any], metric_names: Optional[Sequence[str]] = None
) -> List[str]:
    requested_metrics = metric_names
    if requested_metrics is None:
        requested_metrics = [
            training_cfg["sweep"]["validation_metric"],
            *training_cfg["sweep"].get("evaluation_metrics", []),
        ]

    result: List[str] = []
    for metric_name in requested_metrics:
        if metric_name not in result:
            result.append(metric_name)

    return result


def collect_labels_and_scores(
    detector,
    dataset,
    training_cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = get_dataloader(dataset, {**training_cfg["training"], "shuffle": False})

    detector.eval()

    if isinstance(detector, LSTMPredictionAnomalyDetector):
        subseq_lengths, window_size = _get_subseq_lengths_and_window_size(
            loader.dataset
        )
        labels, scores = detector.get_labels_and_scores(
            loader, subseq_lengths=subseq_lengths, window_size=window_size
        )
    else:
        labels, scores = detector.get_labels_and_scores(loader)

    return labels, scores


def _compute_fbeta(precision: float, recall: float, beta: float = 1.0) -> float:
    if precision == recall == 0:
        return 0.0

    denominator = beta**2 * precision + recall
    if denominator == 0:
        return 0.0

    return float((1 + beta**2) * precision * recall / denominator)


def _evaluate_metric_at_threshold(
    metric_name: str,
    labels: torch.Tensor,
    scores: torch.Tensor,
    threshold: float,
) -> Tuple[str, Dict[str, Any]]:
    evaluator = Evaluator()

    if metric_name in {"best_f1_score", "f1_score"}:
        predictions = torch.greater_equal(scores, threshold).long()
        return "f1_score", {
            "score": float(
                sklearn_f1_score(labels.numpy(), predictions.numpy(), pos_label=1)
            ),
            "info": {"threshold": float(threshold)},
        }

    if metric_name in {"best_ts_f1_score", "ts_f1_score"}:
        predictions = torch.greater(scores, threshold).long()
        precision, recall = ts_precision_and_recall(
            labels,
            predictions,
            alpha=0,
            recall_cardinality_fn=improved_cardinality_fn,
            weighted_precision=True,
        )
        return "ts_f1_score", {
            "score": _compute_fbeta(precision, recall, beta=1.0),
            "info": {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
            },
        }

    if metric_name in {"best_ts_f1_score_classic", "ts_f1_score_classic"}:
        predictions = torch.greater(scores, threshold).long()
        precision, recall = ts_precision_and_recall(
            labels,
            predictions,
            alpha=0,
            recall_cardinality_fn=inverse_proportional_cardinality_fn,
            weighted_precision=False,
        )
        return "ts_f1_score_classic", {
            "score": _compute_fbeta(precision, recall, beta=1.0),
            "info": {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
            },
        }

    score, info = getattr(evaluator, metric_name)(labels, scores)
    return metric_name, {"score": score, "info": info}


def evaluate_metric_set(
    labels: torch.Tensor,
    scores: torch.Tensor,
    training_cfg: Dict[str, Any],
    metric_names: Optional[Sequence[str]] = None,
    *,
    applied_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    evaluator = Evaluator()
    metrics = {}
    validation_metric = training_cfg["sweep"]["validation_metric"]

    for metric_name in _resolve_metric_names(training_cfg, metric_names):
        try:
            if applied_threshold is None:
                score, info = getattr(evaluator, metric_name)(labels, scores)
                report_name = metric_name
                payload = {"score": score, "info": info}
            else:
                report_name, payload = _evaluate_metric_at_threshold(
                    metric_name, labels, scores, applied_threshold
                )

            if report_name not in metrics:
                metrics[report_name] = payload
        except Exception as exc:
            if metric_name == validation_metric:
                raise
            report_name = THRESHOLD_CALIBRATED_METRIC_NAMES.get(metric_name, metric_name)
            if report_name not in metrics:
                metrics[report_name] = {"score": None, "info": {"error": str(exc)}}

    return metrics


def select_threshold(
    labels: torch.Tensor,
    scores: torch.Tensor,
    metric_name: str,
) -> Dict[str, Any]:
    evaluator = Evaluator()
    score, info = getattr(evaluator, metric_name)(labels, scores)
    threshold = info.get("threshold")
    if threshold is None:
        raise ValueError(
            f"Metric {metric_name!r} does not expose a threshold and cannot be used "
            "for holdout calibration."
        )

    return {
        "metric": metric_name,
        "score": score,
        "info": info,
        "threshold": float(threshold),
    }


def score_detector(
    detector,
    dataset,
    training_cfg: Dict[str, Any],
    metric_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    labels, scores = collect_labels_and_scores(detector, dataset, training_cfg)
    metrics = evaluate_metric_set(
        labels,
        scores,
        training_cfg,
        metric_names=metric_names,
    )

    return {
        "metrics": metrics,
        "num_scores": int(scores.shape[0]),
    }


def _resolve_split_index(split: Sequence[float], split_index: Optional[int]) -> int:
    if split_index is not None:
        return split_index
    return 1 if len(split) > 1 else 0


def _compute_metrics(
    training_cfg: Dict[str, Any], cli_cfg: Dict[str, Any], run_dir: Path
) -> Dict[str, Any]:
    eval_dataset_cfg = build_labelled_test_dataset_cfg(
        training_cfg,
        split=cli_cfg.get("split"),
        split_axis=cli_cfg.get("split_axis"),
    )
    datasets = load_split_datasets(eval_dataset_cfg)
    split_index = _resolve_split_index(
        eval_dataset_cfg["split"], cli_cfg.get("split_index")
    )
    eval_ds = datasets[split_index]
    detector = build_detector_from_run(
        run_dir,
        device=training_cfg["training"]["device"],
        prefer_saved_detector=True,
    )
    summary = score_detector(
        detector,
        eval_ds,
        training_cfg,
        metric_names=cli_cfg.get("metric_names"),
    )

    return {
        **summary,
        "dataset": eval_dataset_cfg,
        "split_index": split_index,
    }


def _build_detector_for_holdout(
    run_dir: Path,
    training_cfg: Dict[str, Any],
    fit_detector_on: str,
):
    device = training_cfg["training"]["device"]

    if fit_detector_on == "saved_artifact":
        detector = build_detector_from_run(
            run_dir,
            device=device,
            prefer_saved_detector=True,
        )
        return detector, {"source": "saved_artifact"}

    detector = build_detector_from_run(
        run_dir,
        device=device,
        prefer_saved_detector=False,
    )

    if fit_detector_on == "none":
        return detector, {"source": "none"}

    if fit_detector_on != "training_validation":
        raise ValueError(
            f"Unsupported fit_detector_on value {fit_detector_on!r}. "
            "Expected one of 'training_validation', 'saved_artifact', or 'none'."
        )

    fit_dataset_cfg, _, fit_loader = build_training_validation_loader(training_cfg)
    fit_split_index = _resolve_split_index(fit_dataset_cfg["split"], None)
    fit_detector_if_supported(detector, fit_loader)
    return detector, {
        "source": "training_validation",
        "dataset": fit_dataset_cfg,
        "split_index": fit_split_index,
    }


def evaluate_run_with_holdout(
    run_dir: str | Path,
    run_id: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    split: Optional[List[float]] = None,
    split_axis: Optional[str] = None,
    tune_split_index: int = 0,
    eval_split_index: int = 1,
    threshold_metric: Optional[str] = None,
    fit_detector_on: str = "training_validation",
    metric_names: Optional[List[str]] = None,
    artifact_name: str = "evaluation_summary.json",
    output_dir: Optional[str | Path] = None,
    log_to_mlflow: bool = True,
) -> Dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    training_cfg = load_training_cfg(run_dir)
    threshold_metric = threshold_metric or training_cfg["sweep"]["validation_metric"]

    eval_dataset_cfg, datasets = build_labelled_test_splits(
        training_cfg,
        split=split,
        split_axis=split_axis,
    )
    if len(datasets) < 2:
        raise ValueError(
            "Holdout evaluation requires at least two labelled test splits."
        )
    if tune_split_index == eval_split_index:
        raise ValueError("tune_split_index and eval_split_index must be different.")
    if not 0 <= tune_split_index < len(datasets):
        raise ValueError(
            f"tune_split_index must be in [0, {len(datasets)}), got {tune_split_index}."
        )
    if not 0 <= eval_split_index < len(datasets):
        raise ValueError(
            f"eval_split_index must be in [0, {len(datasets)}), got {eval_split_index}."
        )

    detector, fit_summary = _build_detector_for_holdout(
        run_dir, training_cfg, fit_detector_on
    )

    tune_labels, tune_scores = collect_labels_and_scores(
        detector,
        datasets[tune_split_index],
        training_cfg,
    )
    tuning_metrics = evaluate_metric_set(
        tune_labels,
        tune_scores,
        training_cfg,
        metric_names=metric_names,
    )
    threshold_selection = select_threshold(
        tune_labels,
        tune_scores,
        threshold_metric,
    )

    eval_labels, eval_scores = collect_labels_and_scores(
        detector,
        datasets[eval_split_index],
        training_cfg,
    )
    evaluation_metrics = evaluate_metric_set(
        eval_labels,
        eval_scores,
        training_cfg,
        metric_names=metric_names,
        applied_threshold=threshold_selection["threshold"],
    )

    summary = _sanitize_json(
        {
            "protocol": "holdout_calibration",
            "fit": fit_summary,
            "threshold_metric": threshold_metric,
            "selected_threshold": threshold_selection["threshold"],
            "metrics": evaluation_metrics,
            "num_scores": int(eval_scores.shape[0]),
            "dataset": eval_dataset_cfg,
            "split_index": eval_split_index,
            "tuning": {
                "dataset": eval_dataset_cfg,
                "split_index": tune_split_index,
                "num_scores": int(tune_scores.shape[0]),
                "metrics": tuning_metrics,
                "threshold_selection": threshold_selection,
            },
            "evaluation": {
                "dataset": eval_dataset_cfg,
                "split_index": eval_split_index,
                "num_scores": int(eval_scores.shape[0]),
                "applied_threshold": threshold_selection["threshold"],
            },
        }
    )

    cli_cfg = {
        "run_dir": str(run_dir),
        "run_id": run_id,
        "tracking_uri": tracking_uri,
        "log_to_mlflow": log_to_mlflow,
    }
    resolved_output_dir = run_dir if output_dir is None else Path(output_dir).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = resolved_output_dir / artifact_name
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    _maybe_log_to_mlflow(training_cfg, cli_cfg, summary_path, summary)
    return summary


def _maybe_log_to_mlflow(
    training_cfg: Dict[str, Any],
    cli_cfg: Dict[str, Any],
    summary_path: Path,
    summary: Dict[str, Any],
) -> None:
    if not cli_cfg.get("log_to_mlflow", True):
        return

    run_id = cli_cfg.get("run_id")
    if run_id is None:
        run_id_file = Path(cli_cfg["run_dir"]) / "mlflow_run_id.txt"
        if run_id_file.exists():
            run_id = run_id_file.read_text(encoding="utf-8").strip() or None

    if run_id is None:
        return

    mlflow.set_tracking_uri(
        _resolve_tracking_uri(training_cfg, cli_cfg.get("tracking_uri"))
    )
    active_run = mlflow.active_run()
    if active_run is not None and active_run.info.run_id == run_id:
        log_hydra_run_reference(cli_cfg["run_dir"])
        mlflow.set_tag("hydra_evaluation_summary", str(summary_path.resolve()))
        for metric_name, metric_payload in summary["metrics"].items():
            if metric_payload["score"] is None:
                continue
            mlflow.log_metric(metric_name, float(metric_payload["score"]))
        return

    with mlflow.start_run(run_id=run_id):
        log_hydra_run_reference(cli_cfg["run_dir"])
        mlflow.set_tag("hydra_evaluation_summary", str(summary_path.resolve()))
        for metric_name, metric_payload in summary["metrics"].items():
            if metric_payload["score"] is None:
                continue
            mlflow.log_metric(metric_name, float(metric_payload["score"]))


def evaluate_run(
    run_dir: str | Path,
    run_id: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    split: Optional[List[float]] = None,
    split_axis: Optional[str] = None,
    split_index: Optional[int] = None,
    metric_names: Optional[List[str]] = None,
    artifact_name: str = "evaluation_summary.json",
    output_dir: Optional[str | Path] = None,
    log_to_mlflow: bool = True,
) -> Dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    hydra_config_path = run_dir / ".hydra" / "config.yaml"
    if not hydra_config_path.exists():
        raise FileNotFoundError(
            f"Hydra config file {hydra_config_path} does not exist."
        )

    cli_cfg = {
        "run_dir": str(run_dir),
        "run_id": run_id,
        "tracking_uri": tracking_uri,
        "split": split,
        "split_axis": split_axis,
        "split_index": split_index,
        "metric_names": metric_names,
        "artifact_name": artifact_name,
        "log_to_mlflow": log_to_mlflow,
    }

    training_cfg = to_plain_config(OmegaConf.load(hydra_config_path))
    summary = _sanitize_json(_compute_metrics(training_cfg, cli_cfg, run_dir))

    resolved_output_dir = run_dir if output_dir is None else Path(output_dir).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = resolved_output_dir / artifact_name
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    _maybe_log_to_mlflow(training_cfg, cli_cfg, summary_path, summary)
    return summary


@hydra.main(version_base=None, config_path="configs/evaluate", config_name="default")
def run(cfg) -> Dict[str, Any]:
    cfg = to_plain_config(cfg)
    if cfg["run_dir"] is None:
        raise ValueError(
            "Set run_dir to the Hydra output directory of a finished training run."
        )

    protocol = cfg.get("protocol", "single_split")
    if protocol == "single_split":
        summary = evaluate_run(
            run_dir=cfg["run_dir"],
            run_id=cfg.get("run_id"),
            tracking_uri=cfg.get("tracking_uri"),
            split=cfg.get("split"),
            split_axis=cfg.get("split_axis"),
            split_index=cfg.get("split_index"),
            metric_names=cfg.get("metric_names"),
            artifact_name=cfg["artifact_name"],
            output_dir=HydraConfig.get().runtime.output_dir,
            log_to_mlflow=cfg.get("log_to_mlflow", True),
        )
    elif protocol == "holdout_calibration":
        summary = evaluate_run_with_holdout(
            run_dir=cfg["run_dir"],
            run_id=cfg.get("run_id"),
            tracking_uri=cfg.get("tracking_uri"),
            split=cfg.get("split"),
            split_axis=cfg.get("split_axis"),
            tune_split_index=cfg.get("tune_split_index", 0),
            eval_split_index=cfg.get("eval_split_index", 1),
            threshold_metric=cfg.get("threshold_metric"),
            fit_detector_on=cfg.get("fit_detector_on", "training_validation"),
            metric_names=cfg.get("metric_names"),
            artifact_name=cfg["artifact_name"],
            output_dir=HydraConfig.get().runtime.output_dir,
            log_to_mlflow=cfg.get("log_to_mlflow", True),
        )
    else:
        raise ValueError(f"Unsupported evaluation protocol {protocol!r}.")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
