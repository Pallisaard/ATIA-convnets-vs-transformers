# import sys
# if "google-colab" in sys.modules:
#     -!pip3 install timm=0.5.4-

import torch
from torch import nn
from transformers import AutoModelForImageClassification
import pytorch_lightning as pl

SWIN_MODEL_NAME = "microsoft/swin-base-patch4-window7-224-in22k"
DEFAULT_ROOT_DIR = "checkpoints/"


def get_swin_model(swin_model_name=SWIN_MODEL_NAME,
                   num_classes=10):
    return AutoModelForImageClassification.from_pretrained(
        swin_model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )


class SWIN(pl.LightningModule):
    def __init__(self,
                 name="microsoft/swin-base-patch4-window7-224-in22k",
                 num_classes=10,
                 default_root_dir="checkpoints/"):
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        self.default_root_dir = default_root_dir
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = get_swin_model(name, num_classes=num_classes)

    def forward(self, x):
        outs = self.model(x)
        return outs.logits

    def training_step(self, batch):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        return optimizer

    def validation_step(self, batch):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

def get_swin_trainer(gpus=1,
                     max_epochs=10,
                     callbacks=[]):
    return pl.Trainer(gpus=gpus, max_epochs=max_epochs, callbacks=callbacks)