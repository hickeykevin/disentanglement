#%%

from pytorch_lightning.utilities.seed import seed_everything
from dspirte import SquaresDataModule
from pytorch_lightning.trainer import Trainer
import torch.nn.functional as F
from autoencoder_network import Shape_Encoder, Shape_Decoder
from pytorch_lightning.loggers.wandb import WandbLogger
from callbacks import ImageReconstructionLoggerCallback, MetricsLoggerCallback
import numpy as np
import plotly.express as px
from autoencoder_module import HAELightingModule
import pandas as pd

BATCH_SIZE = 8
NUM_SAMPLES = 9000
LR = 0.0001
MAX_EPOCHS = 5
HLLE_N_NEIGHBORS = 40
SEED = 42

if __name__ == "__main__":
    seed_everything(SEED)
    results_df = pd.DataFrame(columns=["name", "mcc_score", "embedding"])
    
    datamodule = SquaresDataModule(n_samples=NUM_SAMPLES, batch_size=BATCH_SIZE)
    encoder = Shape_Encoder(2, input_channels=1)
    decoder = Shape_Decoder(2, output_channels=1, vae=False)
    lit_autoencoder = HAELightingModule(encoder=encoder, decoder=decoder, lr=LR, hlle_n_neighbors=HLLE_N_NEIGHBORS)

    logger = WandbLogger(project='disentanglement', name='hae_image_log_test')
    callbacks = [
        MetricsLoggerCallback(current_seed=SEED, hae_only=True), 
        ImageReconstructionLoggerCallback(current_seed=SEED, multiple_seeds=False)
    ]

    trainer = Trainer(
        accelerator='gpu',
        max_epochs=MAX_EPOCHS, 
        #fast_dev_run=True,
        logger=logger,
        callbacks=callbacks,
        deterministic=True
        
        )
    trainer.fit(lit_autoencoder, datamodule)
    for row in callbacks[0].results:
            result_row = pd.Series(dict(zip(results_df.columns, row))).to_frame().T
            results_df = pd.concat([results_df, result_row], axis=0)
    logger.log_table(key="results_table", dataframe=results_df)



