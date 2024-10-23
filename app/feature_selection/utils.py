import gower
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm
from sklearn.neighbors import kneighbors_graph as kng
from sklearn.preprocessing import LabelEncoder


def construct_W(X, metric=None, **kwargs):
    k = kwargs['neighbour_size']
    t = kwargs['t_param']
    if metric == 'gower':
        gower_data = gower.gower_matrix(X.to_numpy())
        # The diagonals are by definition 0, which is not helpful for selection
        np.fill_diagonal(gower_data, 99)
        # Todo: figure out why we use 99 above

        nn_indices = np.argpartition(gower_data, k - 1)[:, :k]
        mask = np.zeros(gower_data.shape, bool)
        mask[np.arange(len(gower_data))[:, None], nn_indices] = True
        S = np.where(mask, gower_data, 0)
        S = csr_matrix(S)
    else:
        S = kng(X, k + 1, mode='distance', metric='euclidean')
        """Euclidean distance is only sensible by setting mode = connectivity"""
    S = (-1 * (S * S)) / (2 * t * t)
    S = S.tocsc()
    S = expm(S)
    S = S.tocsr()

    # [1]  M. Belkin and P. Niyogi, “Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering,” Advances in Neural Information Processing Systems,
    # Vol. 14, 2001. Following the paper to make the weights matrix symmetrix we use this method

    bigger = np.transpose(S) > S
    S = S - S.multiply(bigger) + np.transpose(S).multiply(bigger)
    return S
