from typing import Tuple, Optional, List

from torchvision.models import resnet50, ResNet50_Weights
import torch
from torch import nn
from torch import optim
from torchinfo import summary
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

RESNET50 = "resnet50"


def get_resnet_model(name: str, num_classes: int) -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class Resnet50(pl.LightningModule):
    def __init__(self, name: str = RESNET50,
                 num_classes: int = 10,
                 lr: float = 0.005):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = get_resnet_model(name, num_classes)
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


def get_resnet_trainer(gpus: int = 1,
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


def get_info(model):
    test_input = (1, 3, 224, 224)
    return summary(model, test_input, verbose=0)
