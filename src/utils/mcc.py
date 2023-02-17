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

