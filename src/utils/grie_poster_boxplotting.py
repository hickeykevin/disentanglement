#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import FastICA, PCA
from sklearn.utils import check_random_state
from src.data.dsprite import SquaresDataModule
from PIL import Image
import matplotlib
matplotlib.style.use('ggplot')
import plotly.express as px
import pandas as pd



# %%

# MAKE SURE COLUMN NAMES ARE PRETTY
ae_results = pd.read_csv('ae_results.csv' )
hae_results = pd.read_csv('hae_results.csv' )
full_df = pd.concat([ae_results, hae_results])[['name', 'mcc_score']]
full_df.rename(columns={"name": "Method", "mcc_score": "MCC Score"}, inplace=True)
full_df.Method.replace(
    {'ae': "AE", 'hae': "Ours", "hlle": "HLLE", "hlle+ica": "HLLE+ICA", "pca": "PCA"}
    , inplace=True)

# %%
full_df.plot.box(
    by='Method', ylabel="MCC Score", xlabel="Method", 
    positions=[0,1,2,3,4])
plt.title(None)

plt.show()

# %%

fig = px.box(full_df, y='MCC Score', x='Method',  title="MCC Scores Across Various Methods")
fig.update_xaxes(categoryorder='array', categoryarray=['HLLE+ICA', "Ours", "HLLE", "PCA", "AE"])
fig.update_layout(font_size=18)

#%%
import seaborn as sns
fig, ax = plt.subplots()
sns.set_context('poster', font_scale=0.4)
sns.boxplot(x='Method', y="MCC Score", showfliers=False, 
palette="coolwarm", data=full_df, order=['HLLE+ICA', "Ours", "HLLE", "PCA", "AE"])
ax.set_xlabel("Method", size=12)
ax.set_ylabel("MCC Score", size=12)


# %%
