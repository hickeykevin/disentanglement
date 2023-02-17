#%%
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from typing import Dict, List
import torch.nn.functional as F
from sklearn.pipeline import Pipeline
import numpy as np
import pdb

class AELightingModule(pl.LightningModule):
    def __init__(
        self, 
        encoder: nn.Module,
        decoder: nn.Module,
        optimizer: torch.optim.Optimizer,
        ):

        super().__init__()
        self.save_hyperparameters()
        self.optimzer = optimizer
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, encoder: bool = True):
        if encoder:
            return self.encoder(x)
        else:
            return self.decoder(x)

    def configure_optimizers(self):
        return self.optimizer

    def on_train_start(self):
        numpy_data = self.trainer.datamodule.numpy_data # (N x H x W x C)
        self.numpy_data = numpy_data.reshape(numpy_data.shape[0], -1) # -> (N x H*W*C)

    def training_step(self, batch, batch_idx):

        x, idx = batch[0].to(self.device), batch[1].tolist()
        z = self.forward(x, encoder=True)
        x_hat = self.forward(z, encoder=False)
        loss = F.mse_loss(x_hat, x)
                
        log_dict =  {"total_loss": loss}
        self.log_dict(log_dict, on_epoch=True, on_step=True)
        return loss

       


# %%
