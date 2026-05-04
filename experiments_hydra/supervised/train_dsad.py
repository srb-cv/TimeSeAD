from hydra import main as hydra_main
from hydra.core.hydra_config import HydraConfig

from timesead.models.prediction import LSTMS2SPrediction, LSTMS2SPredictionAnomalyDetector
from timesead.models.supervised import DeepSADTS
from timesead.utils.utils import str2cls
from experiments_hydra.utils.config import to_plain_config
import torch

from experiments_hydra.utils import (
    load_best_model_if_available,
    load_dataset,
    save_active_run_id,
    save_final_artifact,
    start_mlflow_run,
    train_model,
)
import os

@hydra_main(
    version_base=None,
    config_path="../configs",
    config_name="dmc/supervised/train_dsad",
)
def run(cfg):
    output_dir = HydraConfig.get().runtime.output_dir
    with start_mlflow_run(cfg) as logger:
        save_active_run_id(output_dir)
        train_ds, val_ds = load_dataset(**to_plain_config(cfg.dataset))
        dataloader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=cfg.training["batch_size"],
            num_workers=cfg.training["num_workers"],
            drop_last=cfg.training["drop_last"],
        )
        model = DeepSADTS(
            train_loader=dataloader,
            n_features=train_ds.num_features,
            n_samples=len(train_ds),
            rep_dim = train_ds.num_features,
            hidden_dims=cfg.model.hidden_dims,
            act=cfg.model.act,
            attn=cfg.model.attn,
            bias=cfg.model.bias,
            stride=cfg.model.stride,
            n_heads=cfg.model.n_heads,
            d_model=cfg.model.d_model,
            pos_encoding=cfg.model.pos_encoding,
            norm=cfg.model.norm,
            epoch_steps=cfg.model.epoch_steps,
            prt_steps=cfg.model.prt_steps,
            verbose=cfg.model.verbose,
            random_state=cfg.model.random_state,
            device=cfg.training.device,
        )
            
        trainer = train_model(
            model,
            train_ds,
            val_ds,
            cfg.training,
            output_dir,
            logger,
            seed=cfg.training.seed,
            )
        
        model = load_best_model_if_available(trainer, model, cfg.training.epochs)

        detector = None
        if cfg.experiment.train_detector:
            window_size = cfg.dataset.pipeline["prediction"]["args"]["window_size"]
            detector = LSTMS2SPredictionAnomalyDetector(model, half_life=2 * window_size).to(cfg.training.device)

        save_final_artifact({"model": model, "detector": detector}, output_dir)
        return {"model": model, "detector": detector}


if __name__ == "__main__":
    run()
