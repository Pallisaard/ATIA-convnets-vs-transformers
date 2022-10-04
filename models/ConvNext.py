# import sys
# if "google-colab" in sys.modules:
#     -!pip3 install timm=0.5.4-

import torch
from torch import nn
from timm.models import create_model
import pytorch_lightning as pl
from torch import optim

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
                 warmup_steps = 1000,):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = get_convnext_model(name, num_classes)
        self.automatic_optimization = False
        self.warmup_steps = warmup_steps
        self.lr = lr

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.LinearLR(optimizer,
                                                start_factor=0,
                                                end_factor=1,
                                                total_iters=self.warmup_steps)
        return [optimizer], [scheduler]
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        
        opt.zero_grad()
        inputs, labels = batch
        outputs = self(inputs).detach()
        loss = self.loss_fn(outputs, labels)
        
        self.manual_backward(loss)
        opt.step()
        sch.step()
        
        self.log("train_loss", loss)
        return loss


    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs).detach()
        loss = self.loss_fn(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss


def get_convnext_trainer(gpus=1,
                         max_epochs=10,
                         callbacks=[]):
    return pl.Trainer(gpus=gpus, max_epochs=max_epochs, callbacks=callbacks)
