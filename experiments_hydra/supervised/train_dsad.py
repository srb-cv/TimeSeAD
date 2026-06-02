import copy
from typing import Any, Dict

from hydra import main as hydra_main
from hydra.core.hydra_config import HydraConfig

from experiments_hydra.utils.config import to_plain_config
from experiments_hydra.utils.dataset import get_data_labels, get_dataloader
from experiments_hydra.utils.post_training import maybe_evaluate_after_training
from timesead.models.supervised import DSADSupervisionAnomalyDetector
from timesead.models.supervised import DeepSADTS, DSADLoss

from experiments_hydra.utils import (
    load_best_model_if_available,
    load_dataset,
    save_active_run_id,
    save_final_artifact,
    start_mlflow_run,
    train_model,
)


def build_center_dataset_cfg(dataset_cfg: Any) -> Dict[str, Any]:
    """Return a normal-only training dataset config for DeepSAD center initialization."""

    center_cfg = copy.deepcopy(to_plain_config(dataset_cfg))
    center_cfg.setdefault("ds_args", {})
    center_cfg["ds_args"]["training"] = True
    center_cfg["ds_args"]["use_unsupervised_training"] = True
    center_cfg["ds_args"]["use_anomalous_as_normal"] = False
    return center_cfg


@hydra_main(
    version_base=None,
    config_path="../configs",
    config_name="dmc/supervised/train_dsad",
)
def run(cfg):
    output_dir = HydraConfig.get().runtime.output_dir
    with start_mlflow_run(cfg, output_dir=output_dir) as logger:
        save_active_run_id(output_dir)
        train_ds, val_ds = load_dataset(**to_plain_config(cfg.dataset))
        center_train_ds, _ = load_dataset(**build_center_dataset_cfg(cfg.dataset))
        center_loader = get_dataloader(center_train_ds, {**cfg.training, "shuffle": False, "drop_last": False})

        model = DeepSADTS(
            train_loader=center_loader,
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
            c = model.get_center()
            criterion = DSADLoss(c=c)
            detector = DSADSupervisionAnomalyDetector(
                model=model,
                criterion=criterion,
                half_life=window_size,
                ).to(cfg.training.device)

        save_final_artifact({"model": model, "detector": detector}, output_dir)
        maybe_evaluate_after_training(cfg, output_dir)
        return {"model": model, "detector": detector}


if __name__ == "__main__":
    run()
