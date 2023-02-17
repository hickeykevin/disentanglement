from typing import List, Dict, Tuple
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
from tqdm import tqdm
from src.utils.utils import instantiate_callbacks, instantiate_loggers
from src.utils.utils import get_pylogger
import pytorch_lightning as pl

log = get_pylogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    train(cfg=cfg)


def train(cfg: DictConfig) -> Tuple[dict, dict]:
    print(cfg)
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating callbacks...")
    callbacks: List[pl.Callback] = instantiate_callbacks(cfg.get("callbacks"))
    for callback in callbacks:
        if hasattr(callback, "_configure_seed") and callable(callback._configure_seed):
            callback._configure_seed(cfg.seed)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    dm: pl.LightningDataModule = instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: pl.LightningModule = instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[pl.loggers.Logger] = instantiate_loggers(cfg.get('logger'))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: pl.Trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    if cfg.get("train"):
        log.info("Begin training!")
        trainer.fit(model=model, datamodule=dm, ckpt_path=cfg.get("ckpt_path"))

if __name__ == "__main__":
    main()



# if __name__ == "__main__":
#     seed_everything(SEED)
#     results_df = pd.DataFrame(columns=["name", "mcc_score", "embedding"])
    
#     datamodule = SquaresDataModule(n_samples=NUM_SAMPLES, batch_size=BATCH_SIZE)
#     encoder = Shape_Encoder(2, input_channels=1)
#     decoder = Shape_Decoder(2, output_channels=1, vae=False)
#     lit_autoencoder = HAELightingModule(encoder=encoder, decoder=decoder, lr=LR, hlle_n_neighbors=HLLE_N_NEIGHBORS)

#     logger = WandbLogger(project='disentanglement', name='grie_poster_hae_single_seed')
#     logger.experiment.watch(encoder, log_freq=5)
#     logger.experiment.watch(decoder, log_freq=5)
#     callbacks = [
#         MetricsLoggerCallback(current_seed=SEED, hae_only=True), 
#         ImageReconstructionLoggerCallback(current_seed=SEED, epoch_level=1)
#     ]

#     trainer = Trainer(
#         accelerator='gpu',
#         max_epochs=MAX_EPOCHS, 
#         #fast_dev_run=True,
#         logger=logger,
#         callbacks=callbacks,
#         deterministic=True
        
#         )
#     trainer.fit(lit_autoencoder, datamodule)
#     for row in callbacks[0].results:
#             result_row = pd.Series(dict(zip(results_df.columns, row))).to_frame().T
#             results_df = pd.concat([results_df, result_row], axis=0)
#     logger.log_table(key="results_table", dataframe=results_df)



