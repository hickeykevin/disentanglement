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
from scipy.spatial import distance_matrix
from torchmetrics.functional import pairwise_euclidean_distance
import pdb
import numpy as np

class HAELightingModule(pl.LightningModule):
    def __init__(
        self, 
        encoder: nn.Module,
        decoder: nn.Module,
        optimizer: torch.optim.Optimizer,
        hlle,
        ica
        ):

        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder
        self.decoder = decoder
        self.hlle = hlle
        self.ica = ica
        self.hlle_ica_pipe = Pipeline(
            [
                ('hlle', self.hlle),
                ('ica', self.ica)
            ]
        )
       

    def forward(self, x, encoder: bool = True):
        if encoder:
            return self.encoder(x)
        else:
            return self.decoder(x)

    def configure_optimizers(self):
        return self.hparams.optimizer(params=self.parameters())

    def on_train_start(self):
        
        numpy_data = self.trainer.datamodule.numpy_data # (N x H x W x C)
        self.numpy_data = numpy_data.reshape(numpy_data.shape[0], -1) # -> (N x H*W*C)
        self.print("[INFO] Training HLLE and fastICA combinaton")
        self.hlle_ica_embedding = self.hlle_ica_pipe.fit_transform(self.numpy_data)
        
        self.print("[INFO] Complete; begin training HAE")
        self.methods = {"hae": self.encoder, "hlle+ica": self.hlle_ica_pipe}

    def training_step(self, batch, batch_idx):

        x, idx = batch[0].to(self.device), batch[1].tolist()
        unique = torch.unique(x, dim=0, sorted=False)
        if unique.size(0) != x.size(0):
            x = unique
            x_numpy = x.detach().cpu().numpy().reshape(unique.size(0), -1)
            corresponding_hlle_ica = torch.tensor(self.hlle_ica_pipe.transform(x_numpy), device=self.device, dtype=torch.float32)
        else:
            corresponding_hlle_ica = torch.tensor(self.hlle_ica_embedding[idx], dtype=torch.float32, device=self.device)
        

        ########### 
        # Encoder
        ############
        z = self.forward(x, encoder=True)
        encoder_loss = self._criterion(z, corresponding_hlle_ica)
        
        ########### 
        # Decoder
        ############        
        x_hat = self.forward(z, encoder=False)
        decoder_loss = self._criterion(x_hat, x)

        #corr_loss = self._corr_loss(x, z) / 100
        loss = decoder_loss + encoder_loss
                
        log_dict =  {
                "encoder_loss": encoder_loss,
                "decoder_loss": decoder_loss,
                #"corr_loss": corr_loss,
                "total_loss": loss
            }
        self.log_dict(log_dict, on_epoch=True, on_step=True)
        return loss

    def on_train_end(self) -> None:
        # instantiate real z values
        self.print("on_train_end gathering all embeddings")
        model_input = torch.tensor(self.numpy_data, device=self.device, dtype=torch.float32).view(
            -1, 
            1,
            self.trainer.datamodule.shape[-2],
            self.trainer.datamodule.shape[-1]
        )
        
        embedding = torch.squeeze(self(model_input)).detach().cpu().numpy() #remove extra dimensions; change to numpy for plotting purposes
        self.final_embeddings = {"hae": embedding, "hlle+ica": self.hlle_ica_embedding}


    def _criterion(self, a: torch.tensor, b: torch.tensor) -> torch.tensor:
        return F.mse_loss(a, b)

    def _isometry_loss(self, E, x, create_graph=True):
        flattend_x = x.view(x.size(0), -1)
        
        def _func_sum(flattend_x):
            x = flattend_x.view(flattend_x.size(0), *self.trainer.datamodule.shape[1:])
            return E(x).view(-1, 1).sum(dim=0)

        J = torch.autograd.functional.jacobian(_func_sum, flattend_x, create_graph=create_graph).permute(1,0,2)
        Jt = torch.transpose(J, -2, -1)
        Jt_J = torch.matmul(Jt, J)
        I = torch.stack(
            [
                torch.eye(self.trainer.datamodule.shape[-1]) for _ in range(flattend_x.size(0))
            ]
        )
        loss = torch.norm(Jt_J - I, p='fro')
        return loss

        
        
    def _corr_loss(self, real_input: torch.tensor, embedding_input: torch.tensor) -> torch.tensor:
        real_distance = pairwise_euclidean_distance(real_input.view(real_input.size(0), -1)).flatten()

        embedding_distance = pairwise_euclidean_distance(embedding_input.view(embedding_input.size(0), -1)).flatten()
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        pearson = cos(real_distance - real_distance.mean(), embedding_distance - embedding_distance.mean())
        # import pdb; pdb.set_trace()
        return pearson
        # flatten correspondinghlleica distance matrix; 16x16 -> 16*16
        # flatten z distance matrix: 16x16 -> 16*16
        # combine into one tensor
        # torch.coeff(combined tesnor) this is returning nan's need to debug


       


# %%
