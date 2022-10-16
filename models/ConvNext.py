# import sys
# if "google-colab" in sys.modules:
#     -!pip3 install timm=0.5.4-
from typing import Tuple, Optional, List

import torch
from torch import nn
from torch import optim
from timm.models import create_model
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger


CONVNEXT_MODEL_NAME = "convnext_base_in22k"
DEFAULT_ROOT_DIR = "checkpoints/"


def get_convnext_model(convnext_model_name: str, num_classes: int = 10):
    return create_model(convnext_model_name,
                        pretrained=True,
                        num_classes=num_classes)


class ConvNext(pl.LightningModule):
    def __init__(self,
                 name: str = CONVNEXT_MODEL_NAME,
                 num_classes: int = 10,
                 lr: float = 0.00001,
                 warmup_steps: int = 1000):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = get_convnext_model(name, num_classes)
        # self.automatic_optimization = False
        self.warmup_steps = warmup_steps
        self.lr = lr

    def configure_optimizers(self) -> optim.AdamW:
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self,
                      batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx: torch.Tensor,
                      dataset_idx: int = 0) -> torch.Tensor:
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)

        self.log("train_loss", loss, logger=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self,
                        batch: Tuple[torch.Tensor, torch.Tensor],
                        batch_idx: torch.Tensor,
                        dataset_idx: int = 0) -> torch.Tensor:
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        acc = torch.eq(outputs.argmax(dim=1), labels).float().mean()
        self.log("val_loss", loss, on_epoch=True, logger=True, sync_dist=True)
        self.log("val_acc", acc, on_epoch=True, logger=True, sync_dist=True)
        return loss


def get_convnext_trainer(gpus: int = 1,
                         max_epochs: int = 10,
                         callbacks: Optional[List[pl.callbacks.Callback]] = None,
                         log_path: str = "logs/"):
    if callbacks is None:
        callbacks = []
    logger = CSVLogger(log_path, name="convnext")

    accel = "gpu" if torch.cuda.is_available() else "cpu"
    gpus = gpus if torch.cuda.is_available() else 1
    return pl.Trainer(accelerator=accel,
                      devices=gpus,
                      max_epochs=max_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      strategy="ddp",
                      enable_progress_bar=False)
