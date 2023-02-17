
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pytorch_lightning import LightningDataModule
import numpy as np
import os
from PIL import Image

from torchvision import transforms as T

class SquaresDataSet(Dataset):
    def __init__(self, n_samples: int):
        super().__init__()
        self.shape = (n_samples, 1, 64, 64)
        self.numpy_data = self._setup_numpy_dataset()
        self.mean = np.mean(self.numpy_data, axis=(0, 1, 2))[0]
        self.std = np.std(self.numpy_data, axis=(0, 1, 2))[0]
        self.transforms = T.Compose(
            [
                T.ToTensor(),
                #T.Normalize(mean=[0.5,], std=[0.5,])
            ]
        )

    def __len__(self):
        return self.numpy_data.shape[0]
        
    def __getitem__(self, idx):
        sample = self.numpy_data[idx, ...]
        sample = self.transforms(sample)
        return sample, idx

    def _setup_numpy_dataset(self):
        self.x_centers = np.random.uniform(low=5., high=58., size=self.shape[0])
        self.y_centers = np.random.uniform(low=5., high=58., size=self.shape[0])
        all_data = []
        for x, y in zip(self.x_centers, self.y_centers):
            x, y = round(x), round(y)
            data = np.zeros(shape=(self.shape[-2], self.shape[-1]), dtype=np.uint8)
            data[(x-3):(x+3), (y-3):(y+3)] = 255
            all_data.append(data)
            
        all_data = np.stack(all_data, axis=0)
        all_data = np.expand_dims(all_data, axis=-1)
        return all_data

class SquaresDataModule(LightningDataModule):
    def __init__(self, n_samples: int, batch_size: int, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.shape = (n_samples, 1, 64, 64)

    def setup(self, stage: str):
        if stage == "fit":
            self.data = SquaresDataSet(n_samples=self.n_samples)
            self.numpy_data = self.data.numpy_data
            self.x_centers = self.data.x_centers
            self.y_centers = self.data.y_centers

    def train_dataloader(self):
        return DataLoader(
            self.data, 
            batch_size=self.batch_size, 
            num_workers=self.hparams.num_workers, 
            shuffle=True, 
            drop_last=True
        )


  





