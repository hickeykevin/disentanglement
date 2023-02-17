#%%
from pytorch_lightning.utilities.seed import seed_everything
from src.data.dsprite import SquaresDataModule
from pytorch_lightning.trainer import Trainer
from src.models.components.autoencoder_network import Shape_Encoder, Shape_Decoder
from src.models.hae import HAELightingModule
from pytorch_lightning.loggers.wandb import WandbLogger
import numpy as np
import pandas as pd
import plotly.express as px
from src.callbacks.callbacks import MetricsLoggerCallback, ImageReconstructionLoggerCallback
import wandb
from src.models.plain_ae import AELightingModule


BATCH_SIZE = 16
NUM_SAMPLES = 6000
LR = 0.0002
MAX_EPOCHS = 10
HLLE_N_NEIGHBORS = 40

if __name__ == "__main__":
    results_df = pd.DataFrame(columns=["seed", "name", "mcc_score", "embedding"])
    random_seeds = np.random.randint(0, 10000, size=8)
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
        lit_autoencoder = AELightingModule(encoder, decoder, lr=LR)
        
        
        callbacks = [
            MetricsLoggerCallback(seed, hae_only=True), 
            ImageReconstructionLoggerCallback(seed, epoch_level=MAX_EPOCHS)
            ]
        
        wandb_logger = WandbLogger(project='disentanglement', group="group_seed", job_type=f"seed={seed}")

        trainer = Trainer(
            accelerator='gpu',
            max_epochs=MAX_EPOCHS, 
            callbacks=callbacks,
            #fast_dev_run=True,
            logger=[wandb_logger],
            deterministic=True
        )
        trainer.fit(lit_autoencoder, datamodule)
        
        # retreive list of [model name, mcc score, embedding] lists from metrics callback
        # add seed value to row, save to dataframe for easy wandb log table and boxplot of mcc scores
        for row in callbacks[0].results:
            row.insert(0, seed)
            result_row = pd.Series(dict(zip(results_df.columns, row))).to_frame().T
            results_df: pd.DataFrame = pd.concat([results_df, result_row], axis=0)
    
    results_df.to_csv("ae_results.csv")
    wandb_logger.log_table(key="results_table", dataframe=results_df)
    boxplot = px.box(results_df, x='name', y='mcc_score', title="MCC Scores over Varying Seeds")
    boxplot.write_html(f"./wandb/latest-run/tmp/mcc_boxplot_plotly_figure.html")
    wandb_logger.experiment.log({"boxplot": boxplot})



        
       