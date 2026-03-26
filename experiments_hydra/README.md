# Hydra + MLflow Pilot Experiments

This package is a pilot migration layer for replacing Sacred-based experiment
management with Hydra and MLflow while keeping the existing TimeSeAD models,
datasets, and trainer unchanged.

Current pilot coverage:

- `reconstruction/train_dense_ae.py`
- `prediction/train_lstm_prediction_filonov.py`
- `generative/vae/train_donut.py`
- `baselines/train_iqr_ad.py`

The configs live under [`configs/`](configs/) in dataset-first layout, for
example:

- `configs/smd/reconstruction/train_dense_ae.yaml`
- `configs/smd/prediction/train_lstm_prediction_filonov.yaml`
- `configs/smd/vae/train_donut.yaml`
- `configs/smd/baselines/train_iqr_ad.yaml`

Each experiment config now declares both:

- `dataset.pipeline` for training
- `dataset.test_pipeline` for evaluation / grid-search scoring
