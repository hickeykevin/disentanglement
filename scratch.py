#%%
import torch
from src.models.components.autoencoder_network import Shape_Encoder
from src.models.isometric_encoder import LightningIsometricEncoder
from src.data.dsprite import SquaresDataModule
from typing import List, Dict, Tuple
from omegaconf import DictConfig
from hydra.utils import instantiate
import pytorch_lightning as pl
from tqdm import tqdm
from src.utils.utils import get_pylogger
import hydra
from torchmetrics.functional import pairwise_euclidean_distance
from torch.utils.data import TensorDataset, DataLoader


log = get_pylogger(__name__)
# python3 single_seed.py model=isometric_encoder trainer=cpu logger=wandb debug=default
# def train(cfg: DictConfig) -> Tuple[dict, dict]:
#     print(cfg)
#     if cfg.get("seed"):
#         pl.seed_everything(cfg.seed, workers=True)

#     log.info(f"Instantiating datamodule <{cfg.data._target_}>")
#     dm: pl.LightningDataModule = instantiate(cfg.data)

#     log.info(f"Instantiating model <{cfg.model._target_}>")
#     model: pl.LightningModule = instantiate(cfg.model)

    
#     log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
#     trainer: pl.Trainer = instantiate(cfg.trainer)

#     if cfg.get("train"):
#         log.info("Begin training!")
#         trainer.fit(model=model, datamodule=dm, ckpt_path=cfg.get("ckpt_path"))


# @hydra.main(version_base=None, config_path="config", config_name="train")
# def main(cfg: DictConfig):
#     train(cfg=cfg)


# if __name__ == "__main__":
#     main() 

# def batch_jacobian(func, x, create_graph=False):
#     #https://discuss.pytorch.org/t/computing-batch-jacobian-efficiently/80771/5
#     #func: Function you want to compute jacobian for. Probably your network
#     #x: Batch of input instances of shape (batch_size, N). if x is an image flatten all dims except batch
#     def _func_sum(x):
#         x = x.view(-1, 1, 64, 64)
#         return func(x).view(-1, 1).sum(dim=0)
#     return torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1,0,2)


#%%
Enc = Shape_Encoder(2, 1)
dm = SquaresDataModule(1000, 10, num_workers=1)
dm.setup('fit')
loader = dm.train_dataloader()


#%%
torch_X = torch.from_numpy(dm.numpy_data).permute(0, -1, -2, -3).float().unique(dim=0)
X = torch.from_numpy(dm.numpy_data.reshape(1000, -1)).float().unique(dim=0)
distances = pairwise_euclidean_distance(X)
distances = torch.where(distances == 0.0, 1e9, distances)
smallest_5_percent = torch.quantile(distances.flatten(), 0.05, interpolation='nearest')
indicies = (distances < smallest_5_percent).nonzero().tolist()
i = [idx[0] for idx in indicies]
j = [idx[1] for idx in indicies]

X_distances = distances[i, j]
#%%
encodings = Enc(torch_X)

#%%
distances_encodings = pairwise_euclidean_distance(encodings)
correspondnig_distance_encodings = distances_encodings[i, j]
#%%
final_result = torch.stack((X_distances, correspondnig_distance_encodings), dim=1).permute(1, 0)
corr = torch.corrcoef(final_result)
print(corr)














#%%
# Test of above with random numbers
torch_X = torch.randn(1000, 1, 64, 64)
X = torch_X.view(1000, -1)
distances = pairwise_euclidean_distance(X)
distances = torch.where(distances == 0.0, 1e9, distances)
smallest_5_percent = distances.flatten().sort()[0][int(1000*1000*0.05)].item()
indicies = (distances < smallest_5_percent).nonzero().tolist()
i = [idx[0] for idx in indicies]
j = [idx[1] for idx in indicies]

X_distances = distances[i, j]
encodings = Enc(torch_X)
distances_encodings = pairwise_euclidean_distance(encodings)
correspondnig_distance_encodings = distances_encodings[i, j]
final_result = torch.stack((X_distances, correspondnig_distance_encodings), dim=1).permute(1, 0)
corr = torch.corrcoef(final_result)






# cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
# pearson = cos(X_distances - X_distances.mean(), correspondnig_distance_encodings - correspondnig_distance_encodings.mean())

# output = batch_jacobian(Enc, X).view(10, 1, 64, 64)
# output_T = torch.transpose(output, -2, -1)
# result = torch.matmul(output_T, output)
# I = torch.stack([torch.eye(64) for _ in range(10)])
# print(torch.norm((result - I), p="fro"))

# %%
