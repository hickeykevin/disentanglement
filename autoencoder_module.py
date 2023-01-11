#%%
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from typing import Dict, List
import torch.nn.functional as F
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import FastICA
from sklearn.pipeline import Pipeline
import numpy as np

class HAELightingModule(pl.LightningModule):
    def __init__(
        self, 
        encoder: nn.Module,
        decoder: nn.Module,
        hlle_n_neighbors: int=25,
        n_latent_factors: int = 2,
        lr: float = 2e-4
        ):

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr

        self.hlle = LocallyLinearEmbedding(
            method='hessian',
            n_neighbors=hlle_n_neighbors, 
            n_components=n_latent_factors,
            eigen_solver='dense'
            )
        self.ica = FastICA(
            n_components=n_latent_factors,
            whiten='warn'
        )
        self.hlle_ica_pipe = Pipeline(
            [
                ('hlle', self.hlle),
                ('ica', self.ica)
            ]
        )
       
        self.save_hyperparameters()


    def forward(self, x, encoder: bool = True):
        if encoder:
            return self.encoder(x)
        else:
            return self.decoder(x)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)

    def on_train_start(self):
        
        numpy_data = self.trainer.datamodule.numpy_data # (N x H x W x C)
        self.numpy_data = numpy_data.reshape(numpy_data.shape[0], -1) # -> (N x H*W*C)
        self.print("[INFO] Training HLLE and fastICA combinaton")
        self.hlle_ica_embedding = self.hlle_ica_pipe.fit_transform(self.numpy_data)
        
        self.print("[INFO] Complete; begin training HAE")

    def training_step(self, batch, batch_idx: int):
        x = batch # assuming that this batch matches original numpy ordering
        bs = x.size(0)
        z = self.forward(x, encoder=True)
        corresponding_hlle_ica = torch.tensor(self.hlle_ica_embedding[(batch_idx*bs) : (batch_idx*bs)+bs], dtype=torch.float32, device=self.device)
        encoder_loss = F.mse_loss(z, corresponding_hlle_ica)
        x_hat = self.forward(z, encoder=False)
        decoder_loss = F.mse_loss(x_hat, x)
        loss = encoder_loss + decoder_loss
        self.log_dict(
            {
                "encoder_loss": encoder_loss,
                "decoder_loss": decoder_loss,
                "total_loss": loss
            },
            on_epoch=True, on_step=True)
        return loss

       

