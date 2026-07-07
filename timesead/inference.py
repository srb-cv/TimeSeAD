from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import OmegaConf

from timesead.models.prediction import LSTMS2SPredictionAnomalyDetector


class FilonovScorer:
    """Inference helper for Filonov LSTM prediction artifacts.

    The scorer expects artifacts produced by
    ``experiments_hydra/prediction/train_lstm_prediction_filonov.py``.
    """

    def __init__(
        self,
        detector: LSTMS2SPredictionAnomalyDetector,
        window_size: int,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.detector = detector.to(self.device).eval()
        self.window_size = int(window_size)
        self.input_dim = self.detector.model.lstm.recurrent_layers[0].input_size

    @classmethod
    def from_run_dir(
        cls,
        run_dir: str | Path,
        device: str | torch.device = "cpu",
    ) -> "FilonovScorer":
        run_dir = Path(run_dir).resolve()
        payload = cls._load_artifact(run_dir, device=device)
        detector = payload.get("detector")
        if detector is None:
            raise RuntimeError(f"Artifact in {run_dir} does not contain a detector.")
        if not isinstance(detector, LSTMS2SPredictionAnomalyDetector):
            raise TypeError(
                "FilonovScorer requires an LSTMS2SPredictionAnomalyDetector, "
                f"got {type(detector)!r}."
            )

        return cls(
            detector=detector,
            window_size=cls._load_window_size(run_dir),
            device=device,
        )

    def compute_score(self, array: np.ndarray) -> np.ndarray:
        if not isinstance(array, np.ndarray):
            raise TypeError("Expected a numpy array with shape (B, D).")

        tensor = torch.as_tensor(array, dtype=torch.float32, device=self.device)
        return self.compute_score_tensor(tensor).detach().cpu().numpy()

    def compute_score_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 2:
            raise ValueError("Expected input tensor with shape (B, D).")

        tensor = tensor.to(device=self.device, dtype=torch.float32)
        sample_count, input_dim = tensor.shape
        if input_dim != self.input_dim:
            raise ValueError(
                f"Expected feature dimension {self.input_dim}, got {input_dim}."
            )

        full_scores = torch.zeros(
            sample_count,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        if sample_count < 2 * self.window_size:
            return full_scores

        inputs, targets = self._build_next_window_pairs(tensor, self.window_size)
        model_inputs = inputs.transpose(0, 1).contiguous()
        model_targets = targets.transpose(0, 1).contiguous()

        with torch.inference_mode():
            score_windows, _, _ = self.detector.compute_online_anomaly_score(
                (model_inputs, model_targets, 0, 0)
            )

        aligned_scores = self._align_score_windows(
            score_windows,
            total_length=sample_count,
            target_offset=self.window_size,
        )
        return aligned_scores

    @staticmethod
    def _load_artifact(
        run_dir: Path,
        device: str | torch.device = "cpu",
    ) -> Dict[str, Any]:
        artifact_path = run_dir / "artifacts" / "final_model.pth"
        if not artifact_path.exists():
            raise FileNotFoundError(f"Expected artifact {artifact_path} does not exist.")

        return torch.load(artifact_path, map_location=device, weights_only=False)

    @staticmethod
    def _load_window_size(run_dir: Path) -> int:
        config_path = run_dir / ".hydra" / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Expected Hydra config {config_path} does not exist.")

        cfg = OmegaConf.load(config_path)
        return int(cfg.dataset.pipeline.prediction.args.window_size)

    @staticmethod
    def _build_next_window_pairs(
        tensor: torch.Tensor,
        window_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair_count = tensor.shape[0] - (2 * window_size) + 1
        if pair_count < 1:
            raise ValueError("Input must contain at least one input/target window pair.")

        windows = tensor.unfold(0, window_size, 1).transpose(1, 2).contiguous()
        return windows[:pair_count], windows[window_size : window_size + pair_count]

    @staticmethod
    def _align_score_windows(
        scores: torch.Tensor,
        total_length: int,
        target_offset: int,
    ) -> torch.Tensor:
        window_size, window_count = scores.shape
        offsets = torch.arange(window_size, device=scores.device).unsqueeze(1)
        starts = torch.arange(window_count, device=scores.device).unsqueeze(0)
        indices = target_offset + starts + offsets
        valid = indices < total_length

        flat_indices = indices[valid].long()
        flat_scores = scores[valid]

        aligned_scores = torch.zeros(
            total_length,
            dtype=scores.dtype,
            device=scores.device,
        )
        counts = torch.zeros_like(aligned_scores)

        aligned_scores.index_add_(0, flat_indices, flat_scores)
        counts.index_add_(0, flat_indices, torch.ones_like(flat_scores))

        covered = counts > 0
        aligned_scores[covered] /= counts[covered]
        return aligned_scores
