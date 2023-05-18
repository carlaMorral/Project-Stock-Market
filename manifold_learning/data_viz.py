import sys
import pandas as pd
import numpy as np

from sklearn.manifold import LocallyLinearEmbedding, Isomap, MDS, SpectralEmbedding
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances
from collections import defaultdict
import distance
from single_linkage import *
import matplotlib.pyplot as plt
from manifold_learning import *


def standarize_data(data):
    """
    Computes Amplitude Scaling of the given time series
    """
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    data_norm = (data - mean) / std
    return data_norm


def read_data():
    with open("./data.csv", "r") as f:
        df = pd.read_csv(f)
        grouped_df = df.groupby('Stock').apply(lambda x: list(x['Price']))
        lengths = grouped_df.apply(len)
        maxlen = max(np.array(lengths.values))
        grouped_df = grouped_df[grouped_df.apply(len) == maxlen]
        data = np.stack(grouped_df.values)
    return data, grouped_df

data, series = read_data()

if __name__=="__main__":
    metric = distance.euclidean
    embedding = mds(metric=metric, d=1)
    data_norm = standarize_data(data)
    x_dim, y_dim = data_norm.shape
    t_col = np.tile(np.arange(y_dim), (x_dim, 1))
    ts_data = np.stack((t_col.flatten(), data_norm.flatten()), axis=1)
    manifold = np.concatenate((np.repeat(embedding, y_dim, axis=0), ts_data), axis=1)[::30]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(manifold[:,0], manifold[:,1], manifold[:,2])
    ax.set_xlim3d(left=-50, right=50)
    ax.set_zlim3d(bottom=-2, top=2)
    plt.show()

 
