from types import SimpleNamespace

import torch

from experiments_hydra.utils.artifacts import save_final_artifact
from experiments_hydra.utils.mlflow import log_hydra_run_reference, save_active_run_id


def test_save_final_artifact_keeps_hydra_dir_canonical(tmp_path):
    payload = {"value": torch.tensor([1.0, 2.0])}

    artifact_path = save_final_artifact(payload, tmp_path)

    assert artifact_path == tmp_path / "artifacts" / "final_model.pth"
    assert artifact_path.exists()
    restored = torch.load(artifact_path)
    assert torch.equal(restored["value"], payload["value"])


def test_log_hydra_run_reference_sets_mlflow_tags(monkeypatch, tmp_path):
    captured = {}

    monkeypatch.setattr(
        "experiments_hydra.utils.mlflow.mlflow.active_run",
        lambda: object(),
    )
    monkeypatch.setattr(
        "experiments_hydra.utils.mlflow.mlflow.set_tags",
        lambda tags: captured.update(tags),
    )

    tags = log_hydra_run_reference(tmp_path)

    assert tags == captured
    assert captured["hydra_run_dir"] == str(tmp_path.resolve())
    assert captured["hydra_artifact_dir"] == str((tmp_path / "artifacts").resolve())
    assert captured["hydra_config_path"] == str((tmp_path / ".hydra" / "config.yaml").resolve())


def test_save_active_run_id_writes_file_and_tags_run(monkeypatch, tmp_path):
    run = SimpleNamespace(info=SimpleNamespace(run_id="run-123"))
    captured = {}

    monkeypatch.setattr(
        "experiments_hydra.utils.mlflow.mlflow.active_run",
        lambda: run,
    )
    monkeypatch.setattr(
        "experiments_hydra.utils.mlflow.mlflow.set_tags",
        lambda tags: captured.update(tags),
    )

    output_path = save_active_run_id(tmp_path)

    assert output_path == tmp_path.resolve() / "mlflow_run_id.txt"
    assert output_path.read_text(encoding="utf-8") == "run-123"
    assert captured["hydra_run_dir"] == str(tmp_path.resolve())
