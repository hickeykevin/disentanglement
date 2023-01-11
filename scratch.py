#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import FastICA, PCA
from sklearn.utils import check_random_state
from dspirte import SquaresDataModule
from PIL import Image

dm = SquaresDataModule(3000, 16)
dm.setup('fit')
X = dm.numpy_data
X = X.reshape(X.shape[0], X.shape[-2]*X.shape[-1])
#%%
# find "distance" values between input data points
# filter for 5% lowest distance values
# calculate coeff scores of above and corresponding hlle embedding representation
## note: think it will be vector of distances vs vector of distances
    ## will thus be np.corrcoef(realvector_distances, embeddingvector_distances)
# note, hlle embedding will be reparamaterized, but skipping for now
# loop over values of k, calculating step 3 for all values of k
from scipy.spatial import distance_matrix
fake_X_index = np.random.choice(X.shape[0], 300, replace=False)
fake_X = X[fake_X_index]
fake_X_distances = distance_matrix(fake_X, fake_X)

# filtering smallest 5% of distances

# store all i,j coordinates and distanve values, filter out all 0 values
results = []
for i in range(fake_X_distances.shape[0]):
    for j in range(fake_X_distances.shape[1]):
        value = fake_X_distances[i, j]
        if value == 0.0:
            continue
        else:
            results.append((i,j,value))

# sort the remaining distance values, keeping (i,j) coordinates
results = sorted(results, key=lambda x: x[-1])

# filter lowest 5% of i,j pair distance values
num = int(len(results) * 0.05) 
results = results[:num]

#store indicies and X distance values
indicies = [(x[0], x[1]) for x in results]
fake_X_distance_values = [x[2] for x in results]

#%%
# embedding values 
for k in range(15, 20):
    hlle = LocallyLinearEmbedding(
        n_neighbors=k,
        method='hessian',
        random_state=42,
        n_components=2,
        eigen_solver='dense'
    )
    hlle_embedding = hlle.fit_transform(fake_X)

    # create distance matrix on embedding representations
    hlle_embedding_distances = distance_matrix(hlle_embedding, hlle_embedding)

    # store the corresponding i,j coordinates embedding distance values
    hlle_distance_results = []
    for i, j in indicies:
        hlle_distance_results.append(hlle_embedding_distances[i, j])
    corr_coef = np.corrcoef(fake_X_distance_values, hlle_distance_results)[1,0]
    print(k, corr_coef)













#%%
# Variables for manifold learning.
hlle_n_neighbors = 30
random_state = 42

plotting_hlle = LocallyLinearEmbedding(
    n_components=2,
    n_neighbors=hlle_n_neighbors,
    method='hessian',
    random_state=random_state
)

hlle = LocallyLinearEmbedding(
    n_neighbors=hlle_n_neighbors,
    method='hessian',
    random_state=random_state,
    n_components=2,
    eigen_solver='dense'
)
fica = FastICA(
    n_components=2,
    whiten='warn',
    random_state=random_state
)




#%% Main method
hlle_embedding = hlle.fit_transform(all_data)
ica_hlle_embedding = fica.fit_transform(hlle_embedding)



