import numpy as np
from dtaidistance import dtw
from numpy import linalg as LA


def euclidean(xi, xj):
    return LA.norm(xj-xi)


def dynamic_time_warping(xi, xj):
    return dtw.distance_fast(xi, xj, use_pruning=True)
