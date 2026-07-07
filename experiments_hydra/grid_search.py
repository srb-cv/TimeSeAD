import copy
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import hydra
import mlflow
import torch
from hydra.core.hydra_config import HydraConfig
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf

from timesead.utils.metadata import PROJECT_ROOT

from experiments_hydra.evaluate import (
    build_detector_from_run,
    build_labelled_test_splits,
    build_training_validation_loader,
    compute_val_splits,
    fit_detector_if_supported,
    load_training_cfg,
    score_detector,
)
from experiments_hydra.utils.config import to_plain_config
from experiments_hydra.utils.sweep_adapter import (
    build_detector_hydra_overrides,
    build_training_hydra_overrides,
    format_hydra_override_strings,
    get_experiment_module,
    load_sweep_spec,
)


@dataclass
class TrainedRunRef:
    training_point: int
    run_id: str
    run_dir: str
    artifact_uri: str
    overrides: Dict[str, Any]


@dataclass
class CandidateResult:
    training_point: int
    detector_point: int
    run_id: str
    run_dir: str
    artifact_uri: str
    training_overrides: Dict[str, Any]
    detector_overrides: Dict[str, Any]
    combined_overrides: Dict[str, Any]
    selection_metric: str
    selection_value: Optional[float]
    validation_metrics: Dict[str, Any]
    validation_num_scores: int


@dataclass
class FoldResult:
    val_fold: int
    splits: List[int]
    val_fold_index: int
    test_fold_indices: List[int]
    best_candidate: Optional[Dict[str, Any]]
    test_scores: Dict[str, List[Dict[str, Any]]]
    all_validation_scores: List[Dict[str, Any]]


@dataclass
class SweepPlan:
    module_name: str
    training_points: List[Dict[str, Any]]
    detector_points: List[Dict[str, Any]]
    selection_metric: str
    evaluation_metrics: List[str]
    mode: str
    config_name_override: Optional[str]
    sweep_source: str


SEARCH_OVERRIDE_KEYS = (
    "training_param_updates",
    "training_param_grid",
    "detector_param_grid",
    "selection_metric",
    "evaluation_metrics",
    "mode",
)

SEARCH_REPLACE_KEYS = {
    "training_param_grid",
    "detector_param_grid",
    "selection_metric",
    "evaluation_metrics",
    "mode",
}


def _recursive_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _recursive_update(base[key], value)
        else:
            base[key] = value
    return base


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


def _metric_to_float(metric_value: Any) -> Optional[float]:
    if metric_value is None:
        return None
    if isinstance(metric_value, torch.Tensor):
        if metric_value.numel() != 1:
            raise ValueError("Expected scalar tensor metric value.")
        return float(metric_value.item())
    return float(metric_value)


def _select_better(candidate: float, best: Optional[float], mode: str) -> bool:
    if best is None:
        return True
    if mode == "max":
        return candidate > best
    return candidate < best


def _resolve_tracking_uri(config_tracking_uri: Optional[str]) -> str:
    if config_tracking_uri:
        return config_tracking_uri
    return f"sqlite:///{PROJECT_ROOT}/mlruns_hydra/mlflow.db"


def _load_hydra_experiment_cfg(config_name: str):
    config_file = Path(PROJECT_ROOT) / "experiments_hydra" / "configs" / f"{config_name}.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"Hydra experiment config {config_file} does not exist.")

    return OmegaConf.load(config_file)


def _extract_runtime_search_overrides(sweep_cfg) -> Dict[str, Any]:
    overrides = {}
    for key in SEARCH_OVERRIDE_KEYS:
        value = to_plain_config(sweep_cfg.get(key))
        if value is not None:
            overrides[key] = value
    return overrides


def _merge_search_cfg(base_search_cfg: Any, runtime_search_overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged_cfg = OmegaConf.create(to_plain_config(base_search_cfg) or {})
    runtime_cfg = to_plain_config(runtime_search_overrides or {})

    for key, value in runtime_cfg.items():
        plain_value = to_plain_config(value)
        if key in SEARCH_REPLACE_KEYS:
            merged_cfg[key] = plain_value
        else:
            current_value = to_plain_config(merged_cfg.get(key))
            if isinstance(current_value, dict) and isinstance(plain_value, dict):
                merged_cfg[key] = OmegaConf.merge(
                    OmegaConf.create(current_value),
                    OmegaConf.create(plain_value),
                )
            else:
                merged_cfg[key] = plain_value

    return to_plain_config(merged_cfg)


def _build_grid_spec_from_search_cfg(search_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "training_param_updates": search_cfg.get("training_param_updates") or {},
        "training_param_grid": search_cfg.get("training_param_grid") or {},
        "detector_param_grid": search_cfg.get("detector_param_grid") or {},
    }


def _resolve_selection_mode(
    user_mode: Optional[str], spec_mode: Optional[str], selection_metric: str
) -> str:
    if user_mode is not None:
        return user_mode
    if spec_mode is not None:
        return spec_mode

    metric_name = selection_metric.lower()
    if any(token in metric_name for token in ("loss", "error", "mse", "mae")):
        return "min"
    return "max"


def _resolve_metric_names(
    selection_metric: str, evaluation_metrics: Optional[Sequence[str]]
) -> List[str]:
    result = [selection_metric]
    for metric_name in evaluation_metrics or []:
        if metric_name not in result:
            result.append(metric_name)
    return result


def _resolve_sweep_plan(cfg) -> SweepPlan:
    if cfg.sweep.experiment_config is not None:
        experiment_cfg = _load_hydra_experiment_cfg(cfg.sweep.experiment_config)
        search_cfg = _merge_search_cfg(
            experiment_cfg.get("sweep", {}),
            _extract_runtime_search_overrides(cfg.sweep),
        )
        sweep_spec = _build_grid_spec_from_search_cfg(search_cfg)
        selection_metric = (
            search_cfg.get("selection_metric")
            or search_cfg.get("validation_metric")
        )
        if selection_metric is None:
            raise ValueError(
                f"Hydra experiment config {cfg.sweep.experiment_config!r} must define "
                "sweep.selection_metric or sweep.validation_metric."
            )
        evaluation_metrics = (
            search_cfg.get("evaluation_metrics")
            or [selection_metric]
        )
        mode = _resolve_selection_mode(
            search_cfg.get("mode"),
            None,
            selection_metric,
        )
        return SweepPlan(
            module_name=to_plain_config(experiment_cfg["experiment"]["entrypoint"]),
            training_points=build_training_hydra_overrides(sweep_spec),
            detector_points=build_detector_hydra_overrides(sweep_spec),
            selection_metric=selection_metric,
            evaluation_metrics=_resolve_metric_names(
                selection_metric, evaluation_metrics
            ),
            mode=mode,
            config_name_override=cfg.sweep.experiment_config,
            sweep_source=cfg.sweep.experiment_config,
        )

    if cfg.sweep.config_path is None:
        raise ValueError(
            "Set either sweep.experiment_config to a new Hydra config name or "
            "sweep.config_path to an existing experiment_configs/*.yml file."
        )

    spec = load_sweep_spec(cfg.sweep.config_path)
    params = spec["params"]
    search_cfg = _merge_search_cfg(
        {
            "training_param_updates": spec.get("training_param_updates", {}),
            "training_param_grid": spec.get("training_param_grid", {}),
            "detector_param_grid": spec.get("detector_param_grid", {}),
            "selection_metric": params.get("selection_metric"),
            "validation_metric": params.get("validation_metric"),
            "evaluation_metrics": params.get("evaluation_metrics"),
            "mode": params.get("mode"),
        },
        _extract_runtime_search_overrides(cfg.sweep),
    )
    sweep_spec = _build_grid_spec_from_search_cfg(search_cfg)
    selection_metric = (
        search_cfg.get("selection_metric")
        or search_cfg.get("validation_metric")
    )
    if selection_metric is None:
        raise ValueError(
            f"Legacy sweep spec {cfg.sweep.config_path!r} must define params.selection_metric "
            "or params.validation_metric."
        )
    evaluation_metrics = (
        search_cfg.get("evaluation_metrics")
        or [selection_metric]
    )
    mode = _resolve_selection_mode(
        search_cfg.get("mode"),
        None,
        selection_metric,
    )
    return SweepPlan(
        module_name=get_experiment_module(params["training_experiment"]),
        training_points=build_training_hydra_overrides(sweep_spec),
        detector_points=build_detector_hydra_overrides(sweep_spec),
        selection_metric=selection_metric,
        evaluation_metrics=_resolve_metric_names(selection_metric, evaluation_metrics),
        mode=mode,
        config_name_override=None,
        sweep_source=Path(cfg.sweep.config_path).name,
    )


def _write_training_registry(path: Path, trained_runs: List[TrainedRunRef]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"trained_runs": [asdict(run) for run in trained_runs]}, handle, indent=2)


def _load_training_registry(path: Path) -> List[TrainedRunRef]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    return [TrainedRunRef(**run_payload) for run_payload in payload["trained_runs"]]


def _make_run_extra_overrides(
    cfg,
    sweep_id: str,
    point_idx: int,
    run_output_dir: Path,
    sweep_source: str,
) -> List[str]:
    extra = [
        f"+experiment.tags.sweep_id={sweep_id}",
        f"+experiment.tags.grid_point={point_idx}",
        f"+experiment.tags.training_point={point_idx}",
        f"+experiment.tags.sweep_source={sweep_source}",
        f"hydra.run.dir={run_output_dir}",
    ]
    if cfg.sweep.tracking_uri is not None:
        extra.append(f"experiment.tracking_uri={cfg.sweep.tracking_uri}")
    extra.extend(cfg.sweep.extra_overrides)
    return extra


def _read_run_id(run_output_dir: Path) -> str:
    run_id_file = run_output_dir / "mlflow_run_id.txt"
    if not run_id_file.exists():
        raise FileNotFoundError(f"Expected MLflow run id file {run_id_file} does not exist.")

    run_id = run_id_file.read_text(encoding="utf-8").strip()
    if not run_id:
        raise RuntimeError(f"MLflow run id file {run_id_file} is empty.")
    return run_id


def _train_candidates_once(
    cfg,
    plan: SweepPlan,
    sweep_id: str,
    sweep_output_dir: Path,
    client: MlflowClient,
) -> List[TrainedRunRef]:
    trained_runs: List[TrainedRunRef] = []
    for idx, point in enumerate(plan.training_points):
        run_output_dir = sweep_output_dir / f"train_{idx:03d}"
        extra = _make_run_extra_overrides(
            cfg, sweep_id, idx, run_output_dir, plan.sweep_source
        )
        extra.extend(
            [
                "++experiment.train_detector=false",
                "++experiment.evaluate_after_training=false",
            ]
        )
        overrides = format_hydra_override_strings(point, extra_overrides=extra)
        cmd = [sys.executable, "-m", plan.module_name]
        if plan.config_name_override is not None:
            cmd.extend(["--config-name", plan.config_name_override])
        cmd.extend(overrides)
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

        run_id = _read_run_id(run_output_dir)
        artifact_uri = client.get_run(run_id).info.artifact_uri
        trained_runs.append(
            TrainedRunRef(
                training_point=idx,
                run_id=run_id,
                run_dir=str(run_output_dir),
                artifact_uri=artifact_uri,
                overrides=point,
            )
        )

    return trained_runs


def _merge_overrides(
    training_overrides: Dict[str, Any], detector_overrides: Dict[str, Any]
) -> Dict[str, Any]:
    merged = copy.deepcopy(training_overrides)
    _recursive_update(merged, detector_overrides)
    return merged


def _get_pipeline_overrides(training_overrides: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    dataset_overrides = training_overrides.get("dataset")
    if not isinstance(dataset_overrides, dict):
        return None

    pipeline_overrides = dataset_overrides.get("pipeline")
    if not isinstance(pipeline_overrides, dict):
        return None

    return copy.deepcopy(pipeline_overrides)


def _evaluate_candidates_for_fold(
    trained_runs: List[TrainedRunRef],
    detector_points: List[Dict[str, Any]],
    splits: Sequence[float],
    val_fold_index: int,
    selection_metric: str,
    metric_names: Sequence[str],
    mode: str,
    fail_on_missing_metric: bool,
    split_axis_override: Optional[str] = None,
) -> Tuple[Optional[CandidateResult], List[CandidateResult]]:
    best_candidate: Optional[CandidateResult] = None
    all_candidate_results: List[CandidateResult] = []
    training_cfg_cache: Dict[str, Dict[str, Any]] = {}
    fit_loader_cache: Dict[str, Any] = {}
    labelled_test_cache: Dict[Tuple[str, Tuple[float, ...], Optional[str]], Tuple[Dict[str, Any], List[Any]]] = {}

    for trained_run in trained_runs:
        if trained_run.run_id not in training_cfg_cache:
            training_cfg_cache[trained_run.run_id] = load_training_cfg(trained_run.run_dir)
        training_cfg = training_cfg_cache[trained_run.run_id]

        if trained_run.run_id not in fit_loader_cache:
            _, _, fit_loader = build_training_validation_loader(training_cfg)
            fit_loader_cache[trained_run.run_id] = fit_loader
        fit_loader = fit_loader_cache[trained_run.run_id]

        cache_key = (
            trained_run.run_id,
            tuple(float(split) for split in splits),
            split_axis_override,
        )
        if cache_key not in labelled_test_cache:
            labelled_dataset_cfg, labelled_splits = build_labelled_test_splits(
                training_cfg,
                split=splits,
                split_axis=split_axis_override,
                pipeline_overrides=_get_pipeline_overrides(trained_run.overrides),
            )
            labelled_test_cache[cache_key] = (labelled_dataset_cfg, labelled_splits)
        _, labelled_splits = labelled_test_cache[cache_key]
        validation_dataset = labelled_splits[val_fold_index]

        for detector_idx, detector_overrides in enumerate(detector_points):
            detector = build_detector_from_run(
                trained_run.run_dir,
                detector_overrides=detector_overrides,
                device=training_cfg["training"]["device"],
                prefer_saved_detector=False,
            )
            fit_detector_if_supported(detector, fit_loader)
            validation_summary = score_detector(
                detector,
                validation_dataset,
                training_cfg,
                metric_names=metric_names,
            )
            metric_value = _metric_to_float(
                validation_summary["metrics"][selection_metric]["score"]
            )
            if metric_value is None and fail_on_missing_metric:
                raise RuntimeError(
                    f"Run {trained_run.run_id} did not produce selection metric {selection_metric!r} "
                    f"for detector point {detector_idx}."
                )

            candidate_result = CandidateResult(
                training_point=trained_run.training_point,
                detector_point=detector_idx,
                run_id=trained_run.run_id,
                run_dir=trained_run.run_dir,
                artifact_uri=trained_run.artifact_uri,
                training_overrides=trained_run.overrides,
                detector_overrides=detector_overrides,
                combined_overrides=_merge_overrides(
                    trained_run.overrides, detector_overrides
                ),
                selection_metric=selection_metric,
                selection_value=metric_value,
                validation_metrics=validation_summary["metrics"],
                validation_num_scores=validation_summary["num_scores"],
            )
            all_candidate_results.append(candidate_result)

            if metric_value is not None and _select_better(
                metric_value,
                None if best_candidate is None else best_candidate.selection_value,
                mode,
            ):
                best_candidate = candidate_result

            del detector

    return best_candidate, all_candidate_results


def _score_candidate_on_test_folds(
    candidate: CandidateResult,
    splits: Sequence[float],
    test_fold_indices: Sequence[int],
    metric_names: Sequence[str],
    split_axis_override: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    if not test_fold_indices:
        return {}

    training_cfg = load_training_cfg(candidate.run_dir)
    _, _, fit_loader = build_training_validation_loader(training_cfg)
    _, labelled_splits = build_labelled_test_splits(
        training_cfg,
        split=splits,
        split_axis=split_axis_override,
        pipeline_overrides=_get_pipeline_overrides(candidate.training_overrides),
    )

    detector = build_detector_from_run(
        candidate.run_dir,
        detector_overrides=candidate.detector_overrides,
        device=training_cfg["training"]["device"],
        prefer_saved_detector=False,
    )
    fit_detector_if_supported(detector, fit_loader)

    test_scores: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for test_fold_index in test_fold_indices:
        test_summary = score_detector(
            detector,
            labelled_splits[test_fold_index],
            training_cfg,
            metric_names=metric_names,
        )
        for metric_name in metric_names:
            payload = test_summary["metrics"][metric_name]
            test_scores[metric_name].append(
                {
                    "score": payload["score"],
                    "info": payload["info"],
                    "test_fold": test_fold_index,
                    "num_scores": test_summary["num_scores"],
                }
            )

    return dict(test_scores)


def _aggregate_final_scores(
    fold_results: Sequence[FoldResult], metric_names: Sequence[str], total_folds: int
) -> Dict[str, float]:
    final_scores = {metric_name: 0.0 for metric_name in metric_names}
    if total_folds <= 0:
        return final_scores

    for fold_result in fold_results:
        for metric_name in metric_names:
            metric_entries = fold_result.test_scores.get(metric_name, [])
            numeric_scores = [
                _metric_to_float(entry["score"])
                for entry in metric_entries
                if entry["score"] is not None
            ]
            if not numeric_scores:
                continue
            final_scores[metric_name] += (
                sum(numeric_scores) / len(numeric_scores) / total_folds
            )

    return final_scores


@hydra.main(version_base=None, config_path="configs/grid_search", config_name="default")
def run(cfg) -> Dict[str, Any]:
    print("Resolved sweep configuration:")
    plan = _resolve_sweep_plan(cfg)
    print(f"Resolved sweep plan contains {len(plan.training_points)} training points and {len(plan.detector_points)} detector points.")
    if cfg.sweep.max_runs is not None:
        print(f"Limiting to max {cfg.sweep.max_runs} runs as specified in sweep.max_runs.")
        plan.training_points = plan.training_points[: cfg.sweep.max_runs]

    tracking_uri = _resolve_tracking_uri(cfg.sweep.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
    sweep_output_dir = (
        Path(cfg.sweep.output_dir) if cfg.sweep.output_dir is not None else hydra_output_dir
    )
    sweep_id = hydra_output_dir.name

    training_registry_path = hydra_output_dir / "trained_runs.json"
    if cfg.sweep.resume_trained_runs_from is not None:
        registry_path = Path(cfg.sweep.resume_trained_runs_from)
        trained_runs = _load_training_registry(registry_path)
        if len(trained_runs) != len(plan.training_points):
            raise ValueError(
                "Loaded training registry size does not match the current training grid size."
            )
        training_registry_reference = str(registry_path)
    else:
        trained_runs = _train_candidates_once(
            cfg,
            plan,
            sweep_id,
            sweep_output_dir,
            client,
        )
        _write_training_registry(training_registry_path, trained_runs)
        training_registry_reference = str(training_registry_path)

    fold_results: List[FoldResult] = []
    if cfg.sweep.run_evaluation:
        print(f"Evaluating candidates on {cfg.sweep.test_folds} validation folds...")
        for val_fold in range(cfg.sweep.test_folds):
            print(f"Evaluating fold {val_fold}...")
            splits, val_fold_index, test_fold_indices = compute_val_splits(
                cfg.sweep.test_folds, val_fold, padding=cfg.sweep.padding
            )
            best_candidate, all_validation_scores = _evaluate_candidates_for_fold(
                trained_runs,
                plan.detector_points,
                splits,
                val_fold_index,
                plan.selection_metric,
                plan.evaluation_metrics,
                plan.mode,
                cfg.sweep.fail_on_missing_metric,
                cfg.sweep.split_axis_override,
            )
            test_scores = (
                {}
                if best_candidate is None
                else _score_candidate_on_test_folds(
                    best_candidate,
                    splits,
                    test_fold_indices,
                    plan.evaluation_metrics,
                    cfg.sweep.split_axis_override,
                )
            )
            print(f"Fold {val_fold} best candidate: {best_candidate}")
            print(f"Fold {val_fold} test scores: {test_scores}")
            fold_results.append(
                FoldResult(
                    val_fold=val_fold,
                    splits=list(splits),
                    val_fold_index=val_fold_index,
                    test_fold_indices=list(test_fold_indices),
                    best_candidate=None
                    if best_candidate is None
                    else asdict(best_candidate),
                    test_scores=test_scores,
                    all_validation_scores=[asdict(result) for result in all_validation_scores],
                )
            )

    final_scores = _aggregate_final_scores(
        fold_results, plan.evaluation_metrics, cfg.sweep.test_folds
    )

    final_selection = None
    if cfg.sweep.run_final_selection:
        best_candidate, all_validation_scores = _evaluate_candidates_for_fold(
            trained_runs,
            plan.detector_points,
            [1],
            0,
            plan.selection_metric,
            plan.evaluation_metrics,
            plan.mode,
            cfg.sweep.fail_on_missing_metric,
            cfg.sweep.split_axis_override,
        )
        final_selection = {
            "splits": [1],
            "val_fold_index": 0,
            "best_candidate": None if best_candidate is None else asdict(best_candidate),
            "all_validation_scores": [asdict(result) for result in all_validation_scores],
            "final_best_params": None
            if best_candidate is None
            else best_candidate.combined_overrides,
        }

    hydra_output_dir.mkdir(parents=True, exist_ok=True)
    fold_results_path = hydra_output_dir / "fold_results.json"
    final_selection_path = hydra_output_dir / "final_best_params.json"
    summary_path = hydra_output_dir / "sweep_summary.json"

    with open(fold_results_path, "w", encoding="utf-8") as handle:
        json.dump(
            _sanitize_json(
                {
                    "fold_results": [asdict(fold_result) for fold_result in fold_results],
                    "final_scores": final_scores,
                }
            ),
            handle,
            indent=2,
        )

    if final_selection is not None:
        with open(final_selection_path, "w", encoding="utf-8") as handle:
            json.dump(_sanitize_json(final_selection), handle, indent=2)

    summary = {
        "trained_runs_registry": training_registry_reference,
        "trained_runs": [asdict(run) for run in trained_runs],
        "fold_results": [asdict(fold_result) for fold_result in fold_results],
        "final_scores": final_scores,
        "final_selection": final_selection,
        "best_run": None if final_selection is None else final_selection["best_candidate"],
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(_sanitize_json(summary), handle, indent=2)

    print(json.dumps(_sanitize_json(summary), indent=2))
    return _sanitize_json(summary)


if __name__ == "__main__":
    run()
