from pathlib import Path

import numpy as np
import pytest
import torch

from timesead.inference import FilonovScorer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / "outputs" / "2026-04-30" / "15-11-21"
ARTIFACT_PATH = RUN_DIR / "artifacts" / "final_model.pth"


def _load_export_payload() -> dict:
    if not ARTIFACT_PATH.exists():
        pytest.skip(f"Exported model artifact is not available: {ARTIFACT_PATH}")

    return torch.load(ARTIFACT_PATH, map_location="cpu", weights_only=False)


def _load_export_model():
    return _load_export_payload()["model"].eval()


def _load_export_detector():
    detector = _load_export_payload()["detector"]
    assert detector is not None
    return detector.eval()


def _load_export_scorer():
    if not ARTIFACT_PATH.exists():
        pytest.skip(f"Exported model artifact is not available: {ARTIFACT_PATH}")

    return FilonovScorer.from_run_dir(RUN_DIR, device="cpu")


def _model_input_dim(model) -> int:
    return model.lstm.recurrent_layers[0].input_size


def _to_model_layout(batch: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(batch).transpose(0, 1).contiguous()


def _score_with_detector(detector, model_input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        scores, _, _ = detector.compute_online_anomaly_score(
            (model_input, target, 0, 0)
        )
    return scores


def test_filonov_scorer_aligns_score_windows_to_full_timeline():
    score_windows = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )

    aligned_scores = FilonovScorer._align_score_windows(
        score_windows,
        total_length=8,
        target_offset=3,
    )

    assert torch.allclose(
        aligned_scores,
        torch.tensor([0.0, 0.0, 0.0, 1.0, 3.0, 5.0, 7.0, 9.0]),
    )


def test_exported_filonov_model_predicts_npz_batch(tmp_path):
    model = _load_export_model()
    batch_size = 3
    window_size = 16
    input_dim = _model_input_dim(model)
    batch = np.random.default_rng(0).normal(
        size=(batch_size, window_size, input_dim)
    ).astype(np.float32)
    npz_path = tmp_path / "prediction_batch.npz"
    np.savez_compressed(npz_path, batch=batch)

    model_input = _to_model_layout(np.load(npz_path)["batch"])

    with torch.no_grad():
        prediction = model((model_input,))

    assert prediction.shape == (window_size, batch_size, input_dim)
    assert torch.isfinite(prediction).all()


def test_exported_filonov_detector_scores_npz_window_batch(tmp_path):
    detector = _load_export_detector()

    batch_size = 3
    window_size = 16
    input_dim = _model_input_dim(detector.model)
    rng = np.random.default_rng(0)
    batch = rng.normal(
        size=(batch_size, window_size, input_dim)
    ).astype(np.float32)
    target_batch = rng.normal(
        size=(batch_size, window_size, input_dim)
    ).astype(np.float32)
    npz_path = tmp_path / "score_batch.npz"
    np.savez_compressed(npz_path, batch=batch, target_batch=target_batch)

    loaded_npz = np.load(npz_path)
    model_input = _to_model_layout(loaded_npz["batch"])
    target = _to_model_layout(loaded_npz["target_batch"])

    scores = _score_with_detector(detector, model_input, target)
    aligned_scores = FilonovScorer._align_score_windows(
        scores,
        total_length=batch_size + window_size - 1,
        target_offset=0,
    )

    assert scores.shape == (window_size, batch_size)
    assert aligned_scores.shape == (batch_size + window_size - 1,)
    assert torch.isfinite(scores).all()
    assert torch.isfinite(aligned_scores).all()


def test_exported_filonov_detector_scores_windowed_npz_feature_batch(tmp_path):
    scorer = _load_export_scorer()

    sample_count = 48
    window_size = scorer.window_size
    input_dim = scorer.input_dim
    features = np.random.default_rng(0).normal(
        size=(sample_count, input_dim)
    ).astype(np.float32)
    npz_path = tmp_path / "feature_batch.npz"
    np.savez_compressed(npz_path, features=features)

    loaded_features = np.load(npz_path)["features"]
    scores = scorer.compute_score(loaded_features)
    score_tensor = scorer.compute_score_tensor(torch.from_numpy(loaded_features))

    expected_batch_size = sample_count - (2 * window_size) + 1
    assert expected_batch_size > 0
    assert scores.shape == (sample_count,)
    assert score_tensor.shape == (sample_count,)
    assert np.all(scores[:window_size] == 0)
    assert torch.all(score_tensor[:window_size] == 0)
    assert np.isfinite(scores).all()
    assert torch.isfinite(score_tensor).all()
