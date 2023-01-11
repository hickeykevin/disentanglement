#%%
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from dspirte import SquaresDataModule
from pytorch_lightning.trainer import Trainer
from autoencoder_network import Shape_Encoder, Shape_Decoder
from autoencoder_module import HAELightingModule
from pytorch_lightning.loggers.wandb import WandbLogger
import numpy as np
import pandas as pd
import plotly.express as px
from callbacks import SingleRunCallback
import wandb
from random import randint


BATCH_SIZE = 8
NUM_SAMPLES = 12000
LR = 2e-5
MAX_EPOCHS = 5
HLLE_N_NEIGHBORS = 40

if __name__ == "__main__":
    results_df = pd.DataFrame(columns=["name", "mcc_score", "embedding"])
    random_seeds = np.random.randint(0, 10000, size=5)
    datamodule = SquaresDataModule(n_samples=NUM_SAMPLES, batch_size=BATCH_SIZE)
    datamodule.setup(stage='fit')
    z_values = np.column_stack(
        [
            datamodule.x_centers,
            datamodule.y_centers
        ]
    )
    
    for seed in random_seeds:
        print(f"[INFO] Training all models using seed = {seed} ")
        seed_everything(seed)
        
        encoder = Shape_Encoder(2, input_channels=1)
        decoder = Shape_Decoder(2, output_channels=1, vae=False)
        lit_autoencoder = HAELightingModule(
            encoder=encoder, 
            decoder=decoder,
            lr=LR, 
            hlle_n_neighbors=HLLE_N_NEIGHBORS)
        callback = SingleRunCallback()
        logger = WandbLogger(project='disentanglement', group="group_seed", job_type=f"seed={seed}")

        trainer = Trainer(
            accelerator='gpu',
            max_epochs=MAX_EPOCHS, 
            callbacks=[callback],
            #fast_dev_run=True,
            logger=logger,
            deterministic=True
        )
        trainer.fit(lit_autoencoder, datamodule)
        for row in callback.results:
            result_row = pd.Series(dict(zip(results_df.columns, row))).to_frame().T
            results_df = pd.concat([results_df, result_row], axis=0)
    logger.log_table(key="results_table", dataframe=results_df)
    boxplot = px.box(results_df, x='name', y='mcc_score', title="MCC Scores over Varying Seeds")
    boxplot.write_html(f"./wandb/latest-run/tmp/mcc_boxplot_plotly_figure.html")
    logger.experiment.log({"boxplot": boxplot})



        
       