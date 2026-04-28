# Experiments Hydra Data Flow

This document explains how data moves through the modern `experiments_hydra`
entrypoints, how windowing is applied, and how training, grid search, and
evaluation are wired together.

## High-Level Flow

The modern Hydra experiments are config-driven orchestration scripts. The
experiment entrypoint usually does not contain dataset-specific logic. Instead,
it reads a Hydra YAML config, delegates dataset construction to shared helpers,
trains a model, writes a canonical local artifact, and optionally evaluates the
run.

The typical single-run flow is:

```text
Hydra config
  -> load_dataset(...)
  -> train_model(...)
  -> load_best_model_if_available(...)
  -> build detector, if configured
  -> save_final_artifact(...)
  -> maybe_evaluate_after_training(...)
```

The grid-search flow is:

```text
Hydra sweep config
  -> resolve training grid and detector grid
  -> train each prediction/reconstruction/generative model once
  -> record trained run ids and run directories
  -> for each outer labelled-test fold:
       -> fit detector on the training-validation split
       -> select the best trained-run + detector candidate on the fold validation slice
       -> score that selected candidate on held-out labelled-test slices
  -> aggregate fold metrics
  -> optionally select final best params on the full labelled-test side
```

## Main Design Patterns

The code uses a few recurring patterns.

`Config-driven factory`: Hydra YAML names classes and constructor arguments.
Helpers use `objspec2constructor(...)` to instantiate datasets, transforms,
models, optimizers, schedulers, trainer hooks, and losses.

`Transform pipeline`: raw datasets are wrapped by ordered transforms. Each
transform receives a parent transform and exposes transformed datapoints. This
is close to a decorator or chain-of-responsibility pattern.

`Thin entrypoint`: files like
`experiments_hydra/prediction/train_lstm_prediction_filonov.py` mostly assemble
shared building blocks. Dataset loading, training, artifact saving, MLflow, and
evaluation live in utility modules.

`Strategy-style evaluation`: post-training evaluation selects a protocol through
config. `single_split` calls `evaluate_run(...)`; `holdout_calibration` calls
`evaluate_run_with_holdout(...)`.

`Run registry`: grid search treats trained model runs as immutable candidates.
It stores their MLflow run ids and Hydra run directories in `trained_runs.json`,
then evaluates detector settings against those saved runs.

## Dataset Objects

The base dataset interface is `timesead.data.dataset.BaseTSDataset`.

Important responsibilities:

- `__len__()` returns the number of independent time-series sequences.
- `seq_len` returns sequence length, either one integer or a list per sequence.
- `num_features` reports feature dimensions.
- `__getitem__(index)` returns `(inputs, targets)` for one sequence.
- `get_default_pipeline()` returns dataset-specific default transforms.

`timesead.data.transforms.DatasetSource` is the first transform wrapper around a
raw dataset. It can restrict the dataset either by sequence index
(`axis="batch"`) or by time index inside each sequence (`axis="time"`).

`timesead.data.transforms.PipelineDataset` is the PyTorch-facing dataset. It
stores the last transform in the chain as `sink_transform`; `__getitem__` simply
asks that sink transform for a datapoint.

## Dataset Loading

`experiments_hydra.utils.dataset.load_dataset(...)` is the central dataset
factory for Hydra experiments.

It performs these steps:

1. Instantiate the raw dataset from `dataset.name` and `dataset.ds_args`.
2. Split the raw dataset with `make_dataset_split(...)`.
3. Choose one pipeline per split. If a single dict is provided, it is replicated
   across all splits.
4. If `use_dataset_pipeline` is true, merge the user pipeline into the dataset's
   `get_default_pipeline()` output.
5. Build the transform chain with `make_pipe_from_dict(...)`.
6. Return one `PipelineDataset` per split.

The split logic lives in `timesead.data.transforms.dataset_source.make_dataset_split`.
For `split_axis: batch`, it partitions independent sequences. For
`split_axis: time`, it partitions the time axis inside every sequence.

For example, the common training config:

```yaml
dataset:
  ds_args:
    training: true
  split: [0.75, 0.25]
  split_axis: time
```

means: load the dataset's training side, then split every time series into a
75% training slice and a 25% validation slice.

## Windowing

Windowing is a transform-level operation. It is not a separate dataset load.
Changing the configured window size changes how datapoints are produced from the
same underlying raw sequence.

For reconstruction/generative experiments, windowing is often explicit:

```yaml
pipeline:
  window:
    class: WindowTransform
    args:
      window_size: 120
  reconstruction:
    class: ReconstructionTargetTransform
    args:
      replace_labels: true
```

`WindowTransform` converts a sequence of length `T` into sliding windows of
length `window_size`. With the default `step_size: 1`, the number of windows is
approximately `T - window_size + 1`. Its output sequence length is the window
size.

For Filonov sequence-to-sequence prediction experiments, the window-like logic
is attached through the prediction target transform:

```yaml
pipeline:
  prediction:
    class: timesead.models.prediction.LSTMS2STargetTransform
    args:
      window_size: 50
      replace_labels: true
test_pipeline:
  prediction:
    class: timesead.models.prediction.LSTMS2STargetTransform
    args:
      window_size: 50
      replace_labels: false
```

The training pipeline usually uses `replace_labels: true` because the model
needs prediction targets. The test pipeline uses `replace_labels: false` so
ground-truth anomaly labels remain available for evaluation.

In grid search, when `dataset.pipeline.*.args.window_size` is swept, the trained
run keeps that value in its Hydra config. Evaluation also applies the training
pipeline override to the labelled test pipeline, so the detector sees test data
with the same window shape as the trained model expects.

## DataLoader Construction

`experiments_hydra.utils.dataset.get_dataloader(...)` wraps a
`PipelineDataset` in `torch.utils.data.DataLoader`.

The important training config fields are:

- `batch_size`: number of datapoints per batch.
- `batch_dim`: where tensors are stacked by the custom `collate_fn`.
- `shuffle`: set by the caller; training uses true, validation/evaluation uses false.
- `drop_last`: whether incomplete batches are dropped.
- `num_workers`: DataLoader workers.

`batch_dim` matters because some TimeSeAD models expect batch-first tensors and
others expect sequence-first tensors.

## Single Training Entrypoint

`experiments_hydra/prediction/train_lstm_prediction_filonov.py` is a thin
Hydra entrypoint for the Filonov LSTM prediction model.

Its responsibilities are:

1. Resolve the Hydra output directory.
2. Start or attach an MLflow run.
3. Save the active MLflow run id into the Hydra directory.
4. Load training and validation datasets via `load_dataset(...)`.
5. Instantiate `LSTMS2SPrediction` from config values.
6. Train with `train_model(...)`.
7. Restore the early-stopping best model if available.
8. Optionally build `LSTMS2SPredictionAnomalyDetector`.
9. Save `{"model": model, "detector": detector}` to the canonical local artifact.
10. Optionally run post-training evaluation.

The detector's default `half_life` is derived from the configured prediction
window:

```text
half_life = 2 * dataset.pipeline.prediction.args.window_size
```

## Training Helper

`experiments_hydra.utils.training.train_model(...)` owns the common PyTorch
training setup.

It performs these steps:

1. Set the random seed.
2. Switch Torch into deterministic or fast mode.
3. Build train and validation DataLoaders.
4. Instantiate optimizer, scheduler, and trainer from config specs.
5. Attach checkpointing under `<hydra_run_dir>/checkpoints`.
6. Attach configured trainer hooks such as early stopping.
7. Instantiate one or more losses.
8. Call `trainer.train(...)`.

The helper returns the trainer so the entrypoint can inspect hooks and restore
the best checkpoint.

## Evaluation Data Semantics

There are two different dataset sides controlled by `dataset.ds_args.training`.

`training: true` loads the training side of the dataset. In normal experiments,
this side is split into train and validation subsets by `dataset.split`.

`training: false` loads the labelled test or attack side of the dataset. This
side is used for metric computation because it contains labels intended for
evaluation.

The important point is that setting `training: false` selects the dataset side
first. A split can still be applied afterwards. For example, holdout calibration
uses:

```text
load labelled test side
  -> split labelled test side into [0.3, 0.7]
  -> tune threshold on split 0
  -> report metrics on split 1
```

## Post-Training Evaluation

`experiments_hydra.utils.post_training.maybe_evaluate_after_training(...)`
dispatches evaluation after a single training run.

If `experiment.evaluate_after_training` is false, it does nothing.

If the configured protocol is `single_split`, it calls `evaluate_run(...)`.
That path loads the labelled test side, selects one split, and evaluates the
saved detector.

If the protocol is `holdout_calibration`, it calls
`evaluate_run_with_holdout(...)`. That path avoids threshold leakage by using
different labelled-test splits for threshold selection and metric reporting.

The holdout protocol does this:

1. Load the training config from `<run_dir>/.hydra/config.yaml`.
2. Rebuild a detector from the saved model artifact.
3. Fit detector state on the training-validation split when
   `fit_detector_on: training_validation`.
4. Load labelled test data with `ds_args.training: false`.
5. Split labelled test data, usually `[0.3, 0.7]`.
6. Select a threshold on `tune_split_index`.
7. Apply that fixed threshold on `eval_split_index`.
8. Write `evaluation_summary.json` to the Hydra run directory.
9. Log metrics and Hydra path references to MLflow.

## Grid Search Flow

`experiments_hydra/grid_search.py` implements the modern version of the legacy
two-level flow.

The key dataclasses are:

- `SweepPlan`: resolved training grid, detector grid, metrics, selection mode,
  and experiment module.
- `TrainedRunRef`: one trained model candidate with run id, run directory,
  artifact URI, and training overrides.
- `CandidateResult`: one evaluated trained-run plus detector-override candidate.
- `FoldResult`: selected candidate and held-out scores for one outer fold.

The main phases are:

1. `_resolve_sweep_plan(...)` reads either a modern Hydra experiment config or a
   legacy sweep YAML and resolves training points and detector points.
2. `_train_candidates_once(...)` launches the experiment entrypoint once per
   training hyperparameter point. Detector training and post-training evaluation
   are disabled during this phase.
3. The run id and Hydra run directory for each trained candidate are stored in
   `trained_runs.json`.
4. `compute_val_splits(...)` builds blocked labelled-test splits for each outer
   validation fold, including optional padding around the validation slice.
5. `_evaluate_candidates_for_fold(...)` fits each detector on the
   training-validation split and scores each candidate on the fold validation
   slice.
6. The best candidate for that fold is selected by `selection_metric` and
   `mode`.
7. `_score_candidate_on_test_folds(...)` scores the selected candidate on the
   held-out labelled-test slices that were not used for validation or padding.
8. `_aggregate_final_scores(...)` averages fold-level held-out scores.
9. Optional final selection evaluates candidates on the full labelled-test side
   with `split: [1]` and writes final best params.

This means the outer grid-search average is obtained from selected candidates
across folds, where each selected candidate may come from a different trained
model hyperparameter point and detector hyperparameter point.

## Artifact And Logging Layout

The canonical local run state is the Hydra output directory.

Important files and folders:

- `.hydra/config.yaml`: resolved training config for the run.
- `checkpoints/`: trainer checkpoints.
- `artifacts/final_model.pth`: canonical saved model and detector payload.
- `mlflow_run_id.txt`: active MLflow run id.
- `evaluation_summary.json`: optional post-training evaluation summary.
- `trained_runs.json`: grid-search registry of trained model candidates.
- `fold_results.json`: grid-search fold metrics.
- `final_best_params.json`: optional final selected params.
- `sweep_summary.json`: grid-search summary.

MLflow stores experiment metadata, params, metrics, and tags that point back to
the Hydra run directory. Large run artifacts should not be duplicated into both
systems unless there is a specific reason.

## Practical Rules

Use `dataset.pipeline` for training transforms and `dataset.test_pipeline` for
label-preserving test transforms.

When adding a new model family, keep the entrypoint thin and put reusable logic
in `experiments_hydra/utils` or `experiments_hydra/evaluate.py`.

When adding a swept window size, make sure the labelled-test pipeline receives
the same shape-affecting override as the trained model.

Do not tune thresholds and report final metrics on the same labelled-test
slice. Use `holdout_calibration` for single runs and the outer-fold logic for
grid search.

If `use_dataset_pipeline: true`, remember that the configured pipeline is merged
into the dataset default pipeline rather than replacing it entirely.
