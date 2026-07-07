# TimeSeAD Data Loading Pipeline

This note describes how data moves from raw dataset files to model-ready batches in TimeSeAD.

## High-Level Flow

The experiment-side data pipeline is:

1. Instantiate a raw dataset class such as `SWaTDataset` or `SMDDataset`.
2. Split the dataset into train/validation/test views.
3. Wrap each split in a transform source.
4. Apply a configured chain of transforms such as subsampling, windowing, and target generation.
5. Expose the transformed view as a `PipelineDataset`.
6. Feed that dataset into a PyTorch `DataLoader` with a custom collate function.

In code, the main entry point is [`load_dataset()`](../../timesead_experiments/utils/dataset_ingredient.py) in [`timesead_experiments/utils/dataset_ingredient.py`](../../timesead_experiments/utils/dataset_ingredient.py).

## 1. Raw Dataset Contract

All dataset classes inherit from [`BaseTSDataset`](dataset.py) in [`timesead/data/dataset.py`](dataset.py).

Each dataset must provide:

- `__len__()`: number of independent time series
- `seq_len`: length of each time series
- `num_features`: feature dimension of each time step
- `__getitem__()`: returns `(inputs, targets)` as tuples of tensors
- `get_default_pipeline()`: default transforms to apply for this dataset

Important detail: datasets typically return whole time series, not pre-windowed samples. windowing happens through transforms.

For example, [`SWaTDataset`](swat_dataset.py) in [`timesead/data/swat_dataset.py`](swat_dataset.py):

- checks whether raw and preprocessed files exist
- optionally preprocesses the raw CSV files
- loads one long multivariate sequence
- standardizes features using training statistics
- returns one input tensor of shape `(T, 51)` and one label tensor of shape `(T,)`

`SWaTDataset.__len__()` returns `1`, because the full training file is treated as a single sequence.

## 2. Dataset Instantiation and Split

[`load_dataset()`](../../timesead_experiments/utils/dataset_ingredient.py) in [`timesead_experiments/utils/dataset_ingredient.py`](../../timesead_experiments/utils/dataset_ingredient.py):

- constructs the dataset from config using `objspec2constructor`
- normalizes the requested split ratios
- creates one split view per ratio using `make_dataset_split()`
- merges the dataset default pipeline with the experiment-specific pipeline
- builds a transform chain for each split

Splitting is implemented in [`timesead/data/transforms/dataset_source.py`](transforms/dataset_source.py).

`DatasetSource` is the source transform. It does not change values by itself; it restricts what part of the dataset is visible.

Two split modes are supported:

- `axis='batch'`: split by number of independent sequences
- `axis='time'`: split each sequence along the time dimension

For long single-series datasets such as SWaT, training scripts usually use `split_axis='time'`, so train and validation are different time ranges from the same sequence.

## 3. Transform Chain

Transforms are defined in [`timesead/data/transforms/`](transforms/) and inherit from [`Transform`](transforms/transform_base.py) in [`transform_base.py`](transforms/transform_base.py).

The design is pull-based:

- the `DataLoader` asks the final dataset for item `i`
- the last transform asks its parent for item `i`
- this continues until the source transform reads from the raw dataset

The transform chain is built by [`make_pipe_from_dict()`](transforms/pipeline_dataset.py) in [`timesead/data/transforms/pipeline_dataset.py`](transforms/pipeline_dataset.py).

The configured order matters. Transforms are applied in dictionary order.

Common transforms include:

- `SubsampleTransform`: downsample a sequence by aggregating neighboring points
- `CacheTransform`: cache transformed datapoints
- `WindowTransform`: convert a full sequence into sliding windows
- `ReconstructionTargetTransform`: use the input sequence itself as the target
- `PredictionTargetTransform`: split a window into an input segment and a prediction target segment

### Windowing

[`WindowTransform`](transforms/window_transform.py) in [`timesead/data/transforms/window_transform.py`](transforms/window_transform.py) is the main step that turns one long time series into many samples.

It:

- computes how many windows exist for each sequence
- maps a flat sample index back to `(sequence_index, window_start)`
- slices both inputs and targets to `[start:end]`

If the source dataset contains one long sequence of length `T` and `window_size = 50`, the number of returned samples is approximately `T - 50 + 1` when `step_size = 1`.

## 4. Target Construction

Target-related transforms live in [`timesead/data/transforms/target_transforms.py`](transforms/target_transforms.py).

Examples:

- `ReconstructionTargetTransform(replace_labels=True)` turns `(inputs, labels)` into `(inputs, inputs)`
- `PredictionTargetTransform(...)` returns an input prefix and a future target suffix
- `OneVsRestTargetTransform(...)` converts labels into binary anomaly targets

This is how the same raw dataset can support reconstruction, prediction, or classification-style training.

## 5. PipelineDataset

[`PipelineDataset`](transforms/pipeline_dataset.py) in [`timesead/data/transforms/pipeline_dataset.py`](transforms/pipeline_dataset.py) is the final dataset object passed to PyTorch.

It simply delegates:

- `__getitem__()` to the last transform in the chain
- `__len__()` to the transformed view
- `seq_len` and `num_features` to the final transform state

At this stage, the dataset no longer represents raw files directly. It represents processed samples after all transforms.

## 6. DataLoader and Batch Layout

[`train_model()`](../../timesead_experiments/utils/training_ingredient.py) in [`timesead_experiments/utils/training_ingredient.py`](../../timesead_experiments/utils/training_ingredient.py) creates the PyTorch `DataLoader` objects.

The project uses a custom [`collate_fn(batch_dim)`](dataset.py) from [`timesead/data/dataset.py`](dataset.py).

That function stacks tensors along a configurable batch dimension:

- `batch_dim = 0` gives the common layout `(B, T, D)`
- `batch_dim = 1` gives sequence-first layout `(T, B, D)`

This matters because some models, especially recurrent ones, expect time-first tensors.

## Example: LSTM Autoencoder on SWaT

The experiment definition in [`timesead_experiments/reconstruction/train_lstm_ae.py`](../../timesead_experiments/reconstruction/train_lstm_ae.py) configures:

- a `WindowTransform(window_size=50)`
- a `ReconstructionTargetTransform(replace_labels=True)`
- `batch_dim = 1`

For `SWaTDataset`, the effective flow is:

1. Load the preprocessed SWaT CSV.
2. Convert labels to binary anomaly labels.
3. Standardize features from training statistics.
4. Apply the dataset default pipeline:
   - `SubsampleTransform(subsampling_factor=5, aggregation='first')`
   - `CacheTransform()`
5. Split the long sequence into train and validation ranges along time.
6. Create sliding windows of length `50`.
7. Replace labels with the input window itself for reconstruction.
8. Batch windows as `(T, B, D)` because `batch_dim = 1`.

## Mental Model

The overall pipeline is:

raw files -> dataset class returns full sequence(s) -> `DatasetSource` applies split -> transforms reshape or relabel samples -> `PipelineDataset` exposes transformed items -> `DataLoader` batches them for training

This separation is what makes the project flexible:

- dataset classes are responsible for reading and basic preprocessing
- transform classes are responsible for shaping training samples
- experiment configs decide which transforms are used for a given model
