import math
import numpy as np

from kmeans import kmeans
from numpy import linalg as LA
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph


SIGMA = 1

def gaussian_metric(xi, xj):
    return math.exp((-LA.norm(xj-xi)**2)/2*(SIGMA**2))


def get_input_graph(data, m=None, eps=None, type=None):
    if type=="gaussian":
        return radius_neighbors_graph(data, radius=eps, mode="distance", metric=gaussian_metric).toarray()
    elif type=="nn":
        return kneighbors_graph(data, n_neighbors=m).toarray()


def spectral_clustering(k, graph, tol=1e-4):
    W = graph
    for i in range(W.shape[0]):
        for j in range(W.shape[0]):
            if W[i,j] != 0: W[j,i] = W[i,j]

    D = np.diag(W.sum(axis=0))
    L = D-W

    vaps, veps = LA.eig(L)
    idx = np.argsort(vaps)[:k]
    v = veps[:,idx]

    return kmeans(v, k, tol)
