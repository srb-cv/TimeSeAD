from hydra import main as hydra_main
from hydra.core.hydra_config import HydraConfig

from timesead.models.common import MSEReconstructionAnomalyDetector
from timesead.models.reconstruction import BasicAE
from experiments_hydra.utils.config import to_plain_config

from experiments_hydra.utils import (
    load_best_model_if_available,
    load_dataset,
    save_active_run_id,
    save_final_artifact,
    start_mlflow_run,
    train_model,
)


@hydra_main(version_base=None, config_path="../configs", config_name="exathlon/reconstruction/train_dense_ae")
def run(cfg):
    output_dir = HydraConfig.get().runtime.output_dir

    with start_mlflow_run(cfg) as logger:
        save_active_run_id(output_dir)
        train_ds, val_ds = load_dataset(**to_plain_config(cfg.dataset))
        model = BasicAE(train_ds.num_features * train_ds.seq_len, cfg.model.z_size * train_ds.seq_len)
        trainer = train_model(model, train_ds, val_ds, cfg.training, output_dir, logger, seed=cfg.training.seed)
        model = load_best_model_if_available(trainer, model, cfg.training.epochs)
        detector = None
        if cfg.experiment.train_detector:
            detector = MSEReconstructionAnomalyDetector(model, batch_first=True).to(cfg.training.device)

        save_final_artifact({"model": model, "detector": detector}, output_dir)
        return {"model": model, "detector": detector}


if __name__ == "__main__":
    run()
