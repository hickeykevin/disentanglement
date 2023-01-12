import torch
import pytorch_lightning as pl
import numpy as np
from mcc import mean_corr_coef_np
import plotly.express as px
from sklearn.decomposition import PCA
from typing import List
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
from train_sklearn_models import return_all_sklearn_embeddings
from torchvision.utils import make_grid


def return_mcc_score(z_values: np.array, embedding: np.array):
        return mean_corr_coef_np(z_values, embedding)

def return_hae_embedding(trainer: pl.trainer.Trainer, pl_module: pl.LightningModule):
    hae_input = torch.tensor(pl_module.numpy_data, device=pl_module.device, dtype=torch.float32).view(
            -1, 
            1,
            trainer.datamodule.shape[-2],
            trainer.datamodule.shape[-1]
        )
        
    hae_embedding = torch.squeeze(pl_module.forward(hae_input)).detach().cpu().numpy() #remove extra dimensions; change to numpy for plotting purposes
    return hae_embedding 

def return_embedding_html(z_values: np.array, embedding: np.array, title: str):
    
    scatter = px.scatter(
        x=z_values[:, 0], 
        y=z_values[:, 1],
        color=embedding[:, 0],
        title=title,
        labels={
            "x": "z2",
            "y": "z1"
        },
        color_continuous_scale='Rainbow'
    )
    model_name = title.split(" ")[0]
    path_to_plotly_html = f"./wandb/latest-run/tmp/{model_name}_plotly_figure.html"
    scatter.write_html(path_to_plotly_html, auto_play=False)
    html = wandb.Html(path_to_plotly_html)

    return html

class ImageReconstructionLoggerCallback(pl.Callback):
    def __init__(self, current_seed: int, epoch_level: int = 1):
        self.seed = current_seed
        self.epoch_level = epoch_level

    def on_train_start(self, trainer, pl_module):
        self.wandb_logger = pl_module.logger
        data_size = trainer.datamodule.shape[0]
        skipping_value = data_size // 16
        static_samples = []
        for i in range(0, data_size, skipping_value):
            sample = trainer.datamodule.data[i]
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
    def __init__(self, current_seed: int=None, hae_only: bool = True)-> None:
            self.results = []
            self.seed = current_seed 
            self.hae_only = hae_only
    
    def on_train_end(self, trainer, pl_module):
        
        # instantiate real z values
        z_values = np.column_stack(
            [
            trainer.datamodule.x_centers,
            trainer.datamodule.y_centers
            ]
        )
        if not self.hae_only:
            print("Training Sklearn models")
            results = return_all_sklearn_embeddings(trainer.datamodule)
        else:
            results = {}
        
        results["hlle+ica"] = pl_module.hlle_ica_embedding
        results['hae'] = return_hae_embedding(trainer, pl_module)

        for name, embedding in results.items():
            row = [name]

            mcc_score = mean_corr_coef_np(z_values, embedding)
            row.append(mcc_score)
            
            scatter_html = return_embedding_html(
                embedding=embedding,
                z_values=z_values,
                title=name
            )
            row.append(scatter_html)
            
            self.results.append(row)


class MultipleSeedsCallback(pl.Callback):
    def __init__(self):
        super().__init__()        
        columns = ["model_name", "mcc_score", "embedding_plot"]
        self.wandb_table = wandb.Table(
            columns=columns
        )

    def on_train_start(self, trainer, pl_module):
        " run all sklearn models with various seeds, "
        random_seeds = range(3)
        # establish all sklearn models models for various seeds
        pca_models = [('pca', PCA(n_components=2, random_state=seed)) for seed in random_seeds]
        
        hlle_ica_update_params = [{'hlle__random_state': seed, 'ica__random_state': seed} for seed in random_seeds]
        hlle_ica_models = [('hlle+ica', deepcopy(pl_module.hlle_ica_pipe).set_params(**param)) for param in hlle_ica_update_params]
        
        all_models = pca_models + hlle_ica_models
        
        # instantiate real z values
        z_values = np.column_stack(
            [
            trainer.datamodule.x_centers,
            trainer.datamodule.y_centers
            ]
        )

        # instantiate squares data as numpy format for sklearn models training
        numpy_data = trainer.datamodule.numpy_data
        self.numpy_data = numpy_data.reshape(numpy_data.shape[0], numpy_data.shape[-2]*numpy_data.shape[-1])
        
        print("[INFO] Training all sklearn models over random seeds")
        for name, model in tqdm(all_models):
            embedding = model.fit_transform(self.numpy_data)
            score = return_mcc_score(z_values, embedding)
            self.wandb_table.add_data(
                name,
                score,
                return_embedding(z_values, embedding, title = f"{name} Embedding; z2")
            )
            
    def on_train_end(self, trainer, pl_module):
        x = self.wandb_table.get_column('model_name')
        y = self.wandb_table.get_column('mcc_score')
        
        boxplot = px.box(x=x, y=y)
        path_to_plotly_html = f"./wandb/latest-run/tmp/boxplot_plotly_figure.html"
        boxplot.write_html(path_to_plotly_html, auto_play=False)
        pl_module.logger.experiment.log(
            {
                "results_table": self.wandb_table,
                "boxplot": wandb.Html(path_to_plotly_html)
            }
        )





class MultipleHAESeeds(pl.Callback):
    def on_train_end(self, trainer, pl_module):
        z_values = np.column_stack(
            [
            trainer.datamodule.x_centers,
            trainer.datamodule.y_centers
            ]
        )
        
        hae_input = torch.tensor(pl_module.numpy_data, device=pl_module.device, dtype=torch.float32).view(
            -1, 
            1,
            trainer.datamodule.shape[-2],
            trainer.datamodule.shape[-1]
        )
        
        hae_embedding = torch.squeeze(pl_module.forward(hae_input)).detach().cpu().numpy() #remove extra dimensions; change to numpy for plotting purposes
        pl_module.mcc_score = mean_corr_coef_np(z_values, hae_embedding)
         