# import sys
# if "google-colab" in sys.modules:
#     -!pip3 install timm=0.5.4-
from typing import Tuple, Optional, List

import torch
from torch import nn
from torch import optim
from transformers import AutoModelForImageClassification
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger


SWIN_MODEL_NAME = "microsoft/swin-base-patch4-window7-224-in22k"
DEFAULT_ROOT_DIR = "checkpoints/"


def get_swin_model(swin_model_name: str = SWIN_MODEL_NAME,
                   num_classes: int = 10):
    return AutoModelForImageClassification.from_pretrained(
        swin_model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )


class SWIN(pl.LightningModule):
    def __init__(self,
                 name: str = "microsoft/swin-base-patch4-window7-224-in22k",
                 num_classes: int = 10,
                 default_root_dir: str = "checkpoints/",
                 lr: float = 0.00001):
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        self.default_root_dir = default_root_dir
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = get_swin_model(name, num_classes=num_classes)
        self.lr = lr

    def configure_optimizers(self) -> optim.AdamW:
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).logits

    def training_step(self,
                      batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx: torch.Tensor,
                      dataset_idx: int = 0) -> torch.Tensor:
        torch.cuda.empty_cache()
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

    def test_step(self,
                  batch: Tuple[torch.Tensor, torch.Tensor],
                  batch_idx: torch.Tensor,
                  dataset_idx: int = 0) -> torch.Tensor:
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        acc = torch.eq(outputs.argmax(dim=1), labels).float().mean()
        self.log("test_loss", loss, on_epoch=True, logger=True, sync_dist=True)
        self.log("test_acc", acc, on_epoch=True, logger=True, sync_dist=True)
        return loss

def get_swin_trainer(gpus: int = 1,
                     max_epochs: int = 10,
                     callbacks: Optional[List[pl.callbacks.Callback]] = None,
                     log_path: str = "logs/"):
    if callbacks is None:
        callbacks = []
    logger = CSVLogger(log_path, name="swin")
    return pl.Trainer(accelerator="gpu",
                      devices=gpus,
                      max_epochs=max_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      strategy="ddp",
                      enable_progress_bar=False)
