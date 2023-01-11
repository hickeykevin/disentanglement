#%%
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

# Credit: https://github.com/ilkhem/icebeem/blob/master/metrics/mcc.py
## NUMPY/SCIPY IMPLEMENTATION; IS USEFUL FOR EVAL PURPOSES ONLY
## AUTHOR ALSO INCLUDES DIFFERENTIABLE IMPLEMENTATIONS, WILL HAVE TO CONSIDER THEM,
## AS WELL AS OUT OF SAMPLE PREDICTIONS
def mean_corr_coef_np(x, y):
    """
    A numpy implementation of the mean correlation coefficient metric.
    :param x: numpy.ndarray
    :param y: numpy.ndarray
    :param method: str, optional
            The method used to compute the correlation coefficients.
                The options are 'pearson' and 'spearman'
                'pearson':
                    use Pearson's correlation coefficient
                'spearman':
                    use Spearman's nonparametric rank correlation coefficient
    :return: float
    """
    d = x.shape[1]
    cc = np.corrcoef(x, y, rowvar=False)[:d, d:]
    cc = np.abs(cc)
    score = cc[linear_sum_assignment(-1 * cc)].mean()
    return score

#%%
# from autoencoder_network import Shape_Encoder, Shape_Decoder
# from dspirte import SquaresDataModule
# from sklearn.manifold import LocallyLinearEmbedding
# from sklearn.decomposition import FastICA
# from sklearn.pipeline import Pipeline

# hlle_n_neighbors: int=15
# n_latent_factors: int = 2

# #%%
# dm = SquaresDataModule(3000, 16)
# enc = Shape_Encoder(2, 1)
# dec = Shape_Decoder(2, 1, vae=False)
# hlle = LocallyLinearEmbedding(
#     method='hessian',
#     n_neighbors=hlle_n_neighbors, 
#     n_components=n_latent_factors,
#     eigen_solver='dense'
# )
# ica = FastICA(
#     n_components=n_latent_factors,
#     whiten='warn'
# )
# hlle_ica_pipe = Pipeline(
#     [
#         ('hlle', hlle),
#         ('ica', ica)
#     ]
# )
# #%%
# # Data setup
# dm.setup('fit')
# raw_X = dm.numpy_data
# X = raw_X.reshape(-1, raw_X.shape[-2]*raw_X.shape[-1])
# z_values = np.column_stack((dm.x_centers, dm.y_centers))

# #%%
# # HLLE+ICA Output
# hlle_ica_output = hlle_ica_pipe.fit_transform(X)
# mean_corr_coef_np(z_values, hlle_ica_output)

# #%%
# # Untrained Encoder output
# enc_input = torch.tensor(raw_X, dtype=torch.float32)
# z = enc(enc_input).detach().cpu().numpy()
# mean_corr_coef_np(z_values, z)


# %%
