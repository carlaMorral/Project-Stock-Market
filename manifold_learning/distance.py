import numpy as np
from dtaidistance import dtw
from numpy import linalg as LA
from scipy.spatial.distance import correlation


def euclidean(xi, xj):
    return LA.norm(xj-xi)

def correlation_distance(xi, xj):
    return correlation(xi, xj)

def dynamic_time_warping(xi, xj):
    return dtw.distance_fast(xi, xj, use_pruning=True)

def pairwise_distance(x, y, ord=2):
    return np.linalg.norm(np.expand_dims(x, 1) - np.expand_dims(y, 0), axis=2)
