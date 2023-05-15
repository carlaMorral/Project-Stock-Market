import numpy as np
from numpy import linalg as LA


def euclidean(xi, xj):
    return LA.norm(xj-xi)
