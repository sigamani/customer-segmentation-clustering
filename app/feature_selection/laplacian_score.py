import numpy as np
import pandas as pd
from scipy.sparse import diags
from sklearn.preprocessing import LabelEncoder

from .utils import construct_W


def lap_score(X, metric=None, **kwargs):
    # if 'W' is not specified, use the default W
    if 'W' not in kwargs.keys():

        if 't_param' not in kwargs.keys():
            t_param = 1
        else:
            t = kwargs['t_param']

        if 'neighbour_size' not in kwargs.keys():
            neighbour_size = 5
        else:
            n = kwargs['neighbour_size']

        W = construct_W(X, metric, t_param=t, neighbour_size=n)

    # construct the affinity matrix W
    else:
        W = kwargs['W']

    # build the diagonal D matrix from affinity matrix W
    D = np.array(W.sum(axis=1))

    L = W

    tmp = np.dot(np.transpose(D), X)
    D = diags(np.transpose(D), [0])
    Xt = np.transpose(X)
    t1 = np.transpose(np.dot(Xt, D.todense()))
    t2 = np.transpose(np.dot(Xt, L.todense()))
    # compute the numerator of Lr
    tmp = np.multiply(tmp, tmp) / D.sum()
    D_prime = np.sum(np.multiply(t1, X), 0) - tmp
    # compute the denominator of Lr
    L_prime = np.sum(np.multiply(t2, X), 0) - tmp
    # avoid the denominator of Lr to be 0
    D_prime[D_prime < 1e-12] = 10000
    # compute laplacian score for all features
    score = 1 - np.array(np.multiply(L_prime, 1 / D_prime))[0, :]
    return np.transpose(score)


"""
    Rank features in ascending order according to their laplacian scores, the smaller the laplacian score is, the more
    important the feature is
"""


def feature_ranking(score):
    idx = np.argsort(score, 0)
    return idx


def LaplacianScore(X, metric=None, **kwargs):
    """
    Method of feature selection outlined in this paper: https://proceedings.neurips.cc/paper/2005/file/b5b03f06271f8917685d14cea7c6c50a-Paper.pdf
    :param X: Dataframe or np.array
    :param metric: Distance metric to be used - str
    :param kwargs: t_param and n_neighbours expected
    :return: np.array of feature importances
    """
    # Todo: Make this into a class with methods
    if 'W' not in kwargs.keys():

        if 't_param' not in kwargs.keys():
            t_param = 1
        else:
            t = kwargs['t_param']

        if 'neighbour_size' not in kwargs.keys():
            neighbour_size = 5
        else:
            n = kwargs['neighbour_size']

        W = construct_W(X, metric=metric, t_param=t, neighbour_size=n)
        n_samples, n_features = np.shape(X)
    # construct the affinity matrix W
    else:
        W = kwargs['W']

    # construct the diagonal matrix
    D = np.array(W.sum(axis=1))
    D = diags(np.transpose(D), [0])
    # construct graph Laplacian L
    L = D - W.toarray()

    # construct 1= [1,···,1]'
    I = np.ones((n_samples, n_features))

    # construct fr' => fr= [fr1,...,frn]'
    if metric == 'gower':
        X = X.apply(LabelEncoder().fit_transform)
    Xt = np.transpose(X)

    # construct fr^=fr-(frt D I/It D I)I
    t = np.matmul(np.matmul(Xt, D.toarray()), I) / np.matmul(np.matmul(np.transpose(I), D.toarray()), I)
    if isinstance(t, pd.DataFrame):
        t = t.to_numpy()
        X = X.to_numpy()
    t = t[:, 0]
    t = np.tile(t, (n_samples, 1))
    fr = X - t

    # Compute Laplacian Score
    fr_t = np.transpose(fr)
    Lr = np.matmul(np.matmul(fr_t, L), fr) / np.matmul(np.dot(fr_t, D.toarray()), fr)

    return np.diag(Lr)
