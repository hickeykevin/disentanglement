#%%
from sklearn.decomposition import PCA
import numpy as np
from typing import List, Dict
from dspirte import SquaresDataModule
import pytorch_lightning as pl


def train_pca_model(X: np.array):
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(X)
    return pca, embedding

def return_all_sklearn_embeddings(datamodule: pl.LightningDataModule) -> Dict:
    '''
    Aggregate all sklearn models name and respective embeddings 

    args:
        datamodule: pytorch lighting datamodule to train on
        funcs: list of seperate model training functions

    returns: Dict[name: embedding] of model name and corressponding embedding array
        
    '''
    funcs = [train_pca_model]
    X = datamodule.numpy_data.reshape(datamodule.numpy_data.shape[0], -1) # (N x H x W x C) -> (N x H*W*C)
    results = {}
    for func in funcs:
        func_name = str(func)
        model_name = func_name.split("_")[1]

        trained_model, embedding = func(X)
        results[model_name] = embedding

    return results

#%%
# dm = SquaresDataModule(3000, 8)
# dm.setup('fit')

# flag = True
# r = return_all_sklearn_embeddings(dm) if flag else {}
# r['hae'] = 5

# # %%
# res = []
# for name, embedding in r.items():
#     row = [name]
#     row.append(embedding)
#     res.append(row)
# %%
