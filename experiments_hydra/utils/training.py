import collections.abc
from pathlib import Path
from typing import Type, Union

import torch
from torch.utils.data import Sampler

from timesead.optim.loss import Loss, TorchLossWrapper
from timesead.optim.trainer import CheckpointHook, EarlyStoppingHook
from timesead.optim.trainer import Trainer
from timesead.utils.rng_utils import set_seed
from timesead.utils.torch_utils import run_deterministic, run_fast
from timesead.utils.utils import objspec2constructor
from timesead.data.sampler import BalancedBatchSampler

from .config import to_plain_config
from .dataset import get_dataloader
from experiments_hydra.utils import get_data_labels
from timesead.models.supervised import DSADLoss


def instantiate_loss(
    loss: Union[str, Loss, Type[Loss], torch.nn.modules.loss._Loss, Type[torch.nn.modules.loss._Loss]]
) -> Loss:
    if isinstance(loss, Loss):
        return loss
    if isinstance(loss, torch.nn.modules.loss._Loss):
        return TorchLossWrapper(loss)

    loss = objspec2constructor(loss)()
    if not isinstance(loss, Loss):
        loss = TorchLossWrapper(loss)

    return loss


def train_model(model, train_ds, val_ds, training_cfg, output_dir: str, logger, seed: int = 0, sampler: Sampler = None):
    training_cfg = to_plain_config(training_cfg)
    set_seed(seed)

    if training_cfg["deterministic"]:
        run_deterministic()
    else:
        run_fast()

    sampler = None
    if "supervised" in training_cfg and training_cfg["supervised"]:
        labels = get_data_labels(
            dataset=train_ds,
        )
        sampler = BalancedBatchSampler(
            labels=labels,
            batch_size=training_cfg["batch_size"]
        )

    train_loader = get_dataloader(train_ds, {**training_cfg, "shuffle": True},sampler)
    val_loader = get_dataloader(val_ds, {**training_cfg, "shuffle": False})

    optimizer = objspec2constructor(training_cfg["optimizer"])
    scheduler = objspec2constructor(training_cfg["scheduler"])
    trainer_ctor = objspec2constructor(training_cfg.get("trainer", Trainer))

    trainer = trainer_ctor(
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device=training_cfg["device"],
        checkpoints=False,
        batch_dimension=training_cfg["batch_dim"],
    )

    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer.add_hook(
        CheckpointHook(out_dir=str(checkpoint_dir), checkpoint_interval=training_cfg["checkpoint_interval"]),
        "post_validation",
    )

    for hook_spec in training_cfg.get("trainer_hooks", []):
        if isinstance(hook_spec, dict):
            event = hook_spec["event"]
            hook = hook_spec["hook"]
        else:
            event, hook = hook_spec
        trainer.add_hook(objspec2constructor(hook)(), event)

    losses = training_cfg["loss"]
    if isinstance(losses, (str, bytes)) or not isinstance(losses, collections.abc.Sequence) or isinstance(losses, dict):
        losses = [losses]

    losses = [instantiate_loss(loss) for loss in losses]
    trainer.train(model, losses, training_cfg["epochs"], log_fn=logger.log_metric)

    return trainer


def load_best_model_if_available(trainer, model, total_epochs: int):
    for hook in reversed(trainer.hooks["post_validation"]):
        if isinstance(hook, EarlyStoppingHook):
            return hook.load_best_model(trainer, model, total_epochs)

    return model
