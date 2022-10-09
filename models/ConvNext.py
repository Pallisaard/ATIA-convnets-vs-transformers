# import sys
# if "google-colab" in sys.modules:
#     -!pip3 install timm=0.5.4-

import torch
from torch import nn
from torch import optim
from timm.models import create_model
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

CONVNEXT_MODEL_NAME = "convnext_base_in22k"
DEFAULT_ROOT_DIR = "checkpoints/"


def get_convnext_model(convnext_model_name, num_classes=10):
    return create_model(convnext_model_name,
                        pretrained=True,
                        num_classes=num_classes)


class ConvNext(pl.LightningModule):
    def __init__(self,
                 name=CONVNEXT_MODEL_NAME,
                 num_classes=10,
                 lr=0.005,
                 warmup_steps = 1000):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = get_convnext_model(name, num_classes)
        # self.automatic_optimization = False
        self.warmup_steps = warmup_steps
        self.lr = lr

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, dataset_idx=0):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)

        self.log("train_loss", loss, logger=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, on_epoch=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, logger=True)
        return loss


def get_convnext_trainer(gpus=1,
                         max_epochs=10,
                         callbacks=[],
                         log_path="logs/"):
    logger = CSVLogger(log_path, name="convnext")

    return pl.Trainer(accelerator="gpu",
                      devices=gpus,
                      max_epochs=max_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      strategy="ddp",
                      enable_progress_bar=False)
