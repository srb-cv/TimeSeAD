from hydra import main as hydra_main
from hydra.core.hydra_config import HydraConfig

from timesead.models.baselines import IQRAnomalyDetector
from experiments_hydra.utils.config import to_plain_config

from experiments_hydra.utils import (
    get_dataloader,
    load_dataset,
    save_active_run_id,
    save_final_artifact,
    start_mlflow_run,
)


@hydra_main(version_base=None, config_path="../configs", config_name="recon/baselines/train_iqr_ad")
def run(cfg):
    output_dir = HydraConfig.get().runtime.output_dir

    with start_mlflow_run(cfg):
        save_active_run_id(output_dir)
        _, val_ds = load_dataset(**to_plain_config(cfg.dataset))
        val_loader = get_dataloader(val_ds, {**to_plain_config(cfg.training), "shuffle": False})

        detector = None
        if cfg.experiment.train_detector:
            detector = IQRAnomalyDetector(
                std_factor=cfg.detector.std_factor,
                first_diffs=cfg.detector.first_diffs,
                cum_method=cfg.detector.cum_method,
                feature_index=cfg.detector.feature_index,
                input_shape=cfg.detector.input_shape,
            ).to(cfg.training.device)
            detector.fit(val_loader)

        save_final_artifact({"model": None, "detector": detector}, output_dir)
        return {"model": None, "detector": detector}


if __name__ == "__main__":
    run()
