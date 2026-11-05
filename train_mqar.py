"""Run the MQAR sweep in mqar_configs.py and save the best model from each run.

Data, model and evaluation come from zoology. Training differs from zoology's
trainer in three ways, the first two taken from KDN's own trainer (pretrain.py):
  1. No weight decay on 1-D parameters or those marked ``_no_weight_decay``
     (such as KDN's A_log, dt_bias and initial_precision_param).
  2. Gradient norm clipping, at 1.0 by default.
  3. The model with the best valid/accuracy is saved. A non-finite loss or
     gradient norm ends the run early.

Environment: MQAR_CKPT_DIR (default: checkpoints), MQAR_GRAD_CLIP, and the options
listed in mqar_configs.py.
"""
import math
import os
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from zoology.config import TrainConfig
from zoology.data.utils import prepare_data
from zoology.logger import WandbLogger
from zoology.model import LanguageModel
from zoology.train import Trainer
from zoology.utils import set_determinism

import mqar_configs


def parameter_groups(model: nn.Module, weight_decay: float) -> list:
    decay, no_decay = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            exempt = getattr(parameter, "_no_weight_decay", False) or parameter.ndim < 2
            (no_decay if exempt else decay).append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class MQARTrainer(Trainer):
    def __init__(self, *args, config: TrainConfig, grad_clip: float, checkpoint_path: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.grad_clip = grad_clip
        self.checkpoint_path = checkpoint_path

    def train_epoch(self, epoch_idx: int) -> None:
        self.model.train()
        iterator = tqdm(self.train_dataloader, desc=f"Train Epoch {epoch_idx}/{self.max_epochs}")
        for inputs, targets, _ in iterator:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            loss, _ = self.compute_loss(inputs, targets)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip).item()
            if not (math.isfinite(loss.item()) and math.isfinite(grad_norm)):
                raise FloatingPointError(f"loss={loss.item()} grad_norm={grad_norm} in epoch {epoch_idx}")
            self.optimizer.step()
            iterator.set_postfix({"loss": loss.item(), "grad_norm": grad_norm})
            self.logger.log({"train/loss": loss.item(), "train/grad_norm": grad_norm, "epoch": epoch_idx})

    def save_checkpoint(self, epoch_idx: int, metrics: dict) -> None:
        temporary = self.checkpoint_path.with_suffix(".tmp")
        torch.save(
            {
                "model": self.model.state_dict(),
                "config": self.config.model_dump(),
                "epoch": epoch_idx,
                "metrics": metrics,
            },
            temporary,
        )
        temporary.replace(self.checkpoint_path)  # A job killed mid-save keeps the previous best.

    def fit(self) -> None:
        self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            parameter_groups(self.model, self.weight_decay),
            lr=self.learning_rate,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_epochs, eta_min=0.0)
        best_accuracy = -1.0
        for epoch_idx in range(self.max_epochs):
            self.train_epoch(epoch_idx)
            metrics = self.test(epoch_idx)
            accuracy = metrics["valid/accuracy"]
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save_checkpoint(epoch_idx, metrics)
                self.logger.log(
                    {"best/epoch": epoch_idx, **{k.replace("valid/", "best/"): v for k, v in metrics.items()}}
                )
            if accuracy > self.early_stopping_threshold:
                print(f"Early stopping at epoch {epoch_idx}: valid/accuracy {accuracy}")
                break
            self.scheduler.step()


def train(config: TrainConfig, checkpoint_dir: Path, grad_clip: float) -> None:
    set_determinism(config.seed)
    logger = WandbLogger(config)
    try:
        logger.log_config(config)
        config.print()
        model = LanguageModel(config.model)
        train_dataloader, test_dataloader = prepare_data(config.data)
        logger.log_model(model, config=config)
        MQARTrainer(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            max_epochs=config.max_epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            early_stopping_metric=config.early_stopping_metric,
            early_stopping_threshold=config.early_stopping_threshold,
            slice_keys=config.slice_keys,
            device="cuda" if torch.cuda.is_available() else "cpu",
            logger=logger,
            config=config,
            grad_clip=grad_clip,
            checkpoint_path=checkpoint_dir / f"{config.run_id}.pt",
        ).fit()
    finally:
        logger.finish()


def main(configs: list) -> int:
    checkpoint_dir = Path(os.environ.get("MQAR_CKPT_DIR", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    grad_clip = float(os.environ.get("MQAR_GRAD_CLIP", 1.0))
    failed = []
    for config in configs:
        try:
            train(config, checkpoint_dir, grad_clip)
        except Exception:
            traceback.print_exc()
            failed.append(config.run_id)
    print(f"Finished {len(configs) - len(failed)}/{len(configs)} runs. Failed: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(mqar_configs.configs))
