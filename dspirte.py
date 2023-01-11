#%%
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pytorch_lightning import LightningDataModule
import numpy as np
import os
from PIL import Image

from torchvision import transforms as T
#%%
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
        return sample

    def _setup_numpy_dataset(self):
        self.x_centers = np.random.uniform(low=5., high=58., size=self.shape[0])
        self.y_centers = np.random.uniform(low=5., high=58., size=self.shape[0])
        all_data = []
        for x, y in zip(self.x_centers, self.y_centers):
            x, y = round(x), round(y)
            data = np.zeros(shape=(self.shape[-2], self.shape[-1]), dtype=np.uint8)
            data[(x-5):(x+5), (y-5):(y+5)] = 255
            all_data.append(data)
            
        all_data = np.stack(all_data, axis=0)
        all_data = np.expand_dims(all_data, axis=-1)
        return all_data

class SquaresDataModule(LightningDataModule):
    def __init__(self, n_samples: int, batch_size: int):
        super().__init__()
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
        return DataLoader(self.data, batch_size=self.batch_size, num_workers=os.cpu_count())


#%%
# dm = SquaresDataModule(3000, 16)
# dm.setup('fit')
# dm.data.__getitem__([0,1])
#%%


# class SquaresDataModule(LightningDataModule):
#     def __init__(self, n_samples: int, batch_size: int):
#         super().__init__()
#         self.batch_size = batch_size
#         self.shape = (n_samples, 1, 64, 64)
    
#     def setup(self, stage: str):
#         if stage == 'fit':
#             self.x_centers = np.random.uniform(low=5., high=58., size=self.shape[0])
#             self.y_centers = np.random.uniform(low=5., high=58., size=self.shape[0])
#             all_data = []
#             for x, y in zip(self.x_centers, self.y_centers):
#                 x = round(x)
#                 y = round(y)
#                 data = torch.zeros(size=(64, 64), dtype=torch.float32)
#                 data[(x-5):(x+5), (y-5):(y+5)] = 255 #make as pixel value
#                 for _ in range(2):
#                     data = data.unsqueeze(dim=0)  #add a batch number and channel number 
#                 all_data.append(data)
                    
#             imgs = torch.cat(all_data)
#             self.data = TensorDataset(imgs)
#             self.numpy_data = imgs.detach().cpu().numpy()

#     def train_dataloader(self):
#         return DataLoader(self.data, batch_size=self.batch_size, num_workers=4) 
#%%

#%%


# class DSpritesDataset(Dataset):
#     def __init__(self, dsprite_imgs: np.array, transform: transforms.Compose):

#         self.imgs = dsprite_imgs
#         self.shape = self.imgs[0].shape
#         self.transform = transform

#     def __len__(self):
#         return len(self.imgs)

#     def __getitem__(self, idx):
#         # Each image in the dataset has binary values so multiply by 255 to get
#         # pixel values
#         sample = self.imgs[idx] * 255
#         # Add extra dimension to turn shape into (H, W) -> (H, W, C)
#         sample = sample.reshape(sample.shape + (1,))

#         if self.transform:
#             sample = self.transform(sample)
#         # Since there are no labels, we just return 0 for the "label" here
#         return sample, 0


# class DSpritesDataModule(LightningDataModule):
#     def __init__(
#         self, 
#         subsample, 
#         transform,
#         batch_size: int=16,
#         data_dir: str='./',
#         path_to_data="https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz", 

#         ):
#         super().__init__()
#         self.subsample=subsample
#         self.transform=transform
#         self.batch_size=batch_size
#         self.data_dir = Path(data_dir)
#         self.path_to_data = path_to_data
#         self.shape = (1, 64, 64)
    
#     def prepare_data(self) -> None:
#         response = requests.get(self.path_to_data)
#         response.raise_for_status()
#         data = np.load(io.BytesIO(response.content))
#         latents_values = data['latents_values'][::self.subsample]
#         imgs = data['imgs'][::self.subsample]
#         imgs = imgs[
#             (latents_values[:, 0] == 1) &\
#             (latents_values[:, 1] == 1) &\
#             (latents_values[:, 2] == 0.5) &\
#             (latents_values[:, 3] == 0.)
#         ]

#         np.save(self.data_dir / "dsprite_imgs", imgs)
#         np.save(self.data_dir / "dsprite_latents", latents_values)

#     def setup(self, stage):
#         self.imgs = np.load(self.data_dir / "dsprite_imgs.npy")
#         self.latents = np.load(self.data_dir / "dsprite_latents.npy")
#         if stage == 'fit':
#             self.dataset = DSpritesDataset(self.imgs, transform=self.transform)
     
#     def train_dataloader(self):
#         return DataLoader(self.dataset, batch_size=self.batch_size)

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((32, 32)) #needed for 64x64 image sizes; think about fixing
# ])

# datamodule = DSpritesDataModule(
#     subsample=1,
#     transform=transform,
# )

# datamodule.prepare_data()
# datamodule.setup(stage='fit')
#%%
         





