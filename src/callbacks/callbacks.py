import torch
import pytorch_lightning as pl
import numpy as np
from src.utils.mcc import mean_corr_coef_np
import plotly.express as px
from sklearn.decomposition import PCA
from typing import Dict, List
import wandb
from tqdm import tqdm
from copy import deepcopy
from torchvision.utils import make_grid
import logging
from torch import nn
import pandas as pd


def return_mcc_score(z_values: np.array, embedding: np.array):
        return mean_corr_coef_np(z_values, embedding)

def return_pytorch_embedding(trainer: pl.trainer.Trainer, pl_module: pl.LightningModule):
    model_input = torch.tensor(pl_module.numpy_data, device=pl_module.device, dtype=torch.float32).view(
            -1, 
            1,
            trainer.datamodule.shape[-2],
            trainer.datamodule.shape[-1]
        )
        
    embedding = torch.squeeze(pl_module.forward(model_input)).detach().cpu().numpy() #remove extra dimensions; change to numpy for plotting purposes
    return embedding 

def return_embedding_html(z_values: np.array, embedding: np.array, latent_choice: int, title: str):
    
    scatter = px.scatter(
        x=z_values[:, 0], 
        y=z_values[:, 1],
        color=embedding[:, latent_choice],
        title=title,
        labels={
            "x": "z1",
            "y": "z2"
        },
        color_continuous_scale='Rainbow'
    )
    path_to_plotly_html = f"./wandb/latest-run/tmp/{title}_plotly_figure.html"
    scatter.write_html(path_to_plotly_html, auto_play=False)
    html = wandb.Html(path_to_plotly_html)

    return html

class ImageReconstructionLoggerCallback(pl.Callback):
    def __init__(self, epoch_level: int = 1):
        self.epoch_level = epoch_level

    def _configure_seed(self, seed: int):
        self.seed = seed

    def on_train_start(self, trainer, pl_module):
        self.wandb_logger = pl_module.logger
        data_size = trainer.datamodule.shape[0]
        skipping_value = data_size // 16
        static_samples = []
        for i in range(0, data_size, skipping_value):
            sample = trainer.datamodule.data[i][0]
            static_samples.append(sample)

        self.static_samples = torch.stack(static_samples, dim=0).to(pl_module.device)

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch+1) % self.epoch_level == 0:
            with torch.no_grad():
                #import pdb; pdb.set_trace()
                z = pl_module.forward(self.static_samples, encoder=True)
                reconstructions = pl_module.forward(z, encoder=False)

            results = []
            for (img, recon) in zip(self.static_samples, reconstructions):
                results.append(img)
                results.append(recon)

            grid = make_grid(results, nrow=2)

            log_grid = wandb.Image(grid, caption="Left: Reference, Right: Reconstruction")

            self.wandb_logger.experiment.log({f"Reconstructions seed={self.seed}": log_grid})


class MetricsLoggerCallback(pl.Callback):
    def __init__(self)-> None:
        self.results = []
    
    def _configure_seed(self, seed):
        self.seed = seed
    
    # not sure if this is running, need to ensure validation step runs or can get correct ordering
    def on_validation_end(self, trainer, pl_module):
        print("callback.on_validaton_end saving metrics and embeddings")
        embeddings = pl_module.final_embeddings

        # instantiate real z values
        z_values = np.column_stack(
            [
            trainer.datamodule.x_centers,
            trainer.datamodule.y_centers
            ]
        )
        # save seed, model name, mcc score and embedding to self.results
        # for wandb logging 
        for name, embedding in embeddings.items():
            row = [self.seed, name]
            mcc_score = mean_corr_coef_np(z_values, embedding)
            row.append(mcc_score)
            scatter_html = return_embedding_html(
                embedding=embedding,
                z_values=z_values,
                title=name
            )
            row.append(scatter_html)
            self.results.append(row)

        column_names = ["seed", "name", "mcc_score", "example_embedding"]
        results_df = pd.DataFrame(columns=column_names)
        
        for row in self.results:
             result_row = pd.Series(dict(zip(results_df.columns, row))).to_frame().T
             results_df = pd.concat([results_df, result_row], axis=0)
        logger.log_table(key="results_table", dataframe=results_df)

        pl_module.logger.log_table()