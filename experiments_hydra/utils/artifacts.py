from pathlib import Path
from typing import Any

import mlflow
import torch


def save_final_artifact(payload: Any, output_dir: str, filename: str = "final_model.pth") -> Path:
    artifact_dir = Path(output_dir) / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = artifact_dir / filename
    torch.save(payload, artifact_path)
    mlflow.log_artifact(str(artifact_path), artifact_path="artifacts")
    return artifact_path

