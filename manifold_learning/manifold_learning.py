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

def dist_to_kernel(D, sigma=.1):
    return np.exp(-D*D/(2*sigma*sigma))

def isomap(metric, k=20, d=1):
    data_norm = standarize_data(data)
    isomap = Isomap(n_neighbors=k, n_components=d, metric=metric).fit(data_norm)
    return isomap.embedding_


def lle(metric=None, k=20, d=1):
    data_norm = standarize_data(data)
    lle = LocallyLinearEmbedding(n_neighbors=k, n_components=d, random_state=0).fit(data_norm)
    return lle.embedding_


def mds(metric, d=1):
    data_norm = standarize_data(data)
    dist_matrix = pairwise_distances(X=data_norm, metric=metric)
    mds = MDS(n_components=d, dissimilarity="precomputed", random_state=0).fit(dist_matrix)
    return mds.embedding_
    
def laplacian(metric, d=1):
    data_norm = standarize_data(data)
    aff_matrix = dist_to_kernel(pairwise_distances(X=data_norm, metric=metric), sigma=50)
    laplacian = SpectralEmbedding(n_components=d, affinity="precomputed", random_state=0).fit(aff_matrix)
    return laplacian.embedding_

def distortion(metric, embedding):
    data_norm = standarize_data(data)
    dist1 = pairwise_distances(X=data_norm, metric=metric)
    dist2 = pairwise_distances(X=embedding)
    return np.abs(dist1 - dist2).mean(), dist1.mean()

if __name__=="__main__":
    d = []
    for i in range(1, 9):
        metric = distance.euclidean
        embedding = isomap(metric=metric, d=i)
        d1, d0 = distortion(metric, embedding)
        if len(d) == 0:
            d.append(d0)
        d.append(d1)
    plt.title('Isomap embeddings with Euclidean distance')
    plt.xlabel('dimensions')
    plt.ylabel('distortion')
    plt.plot(range(9),d)
    plt.show()
    
    d = []
    for i in range(1, 9):
        metric = distance.correlation_distance
        embedding = isomap(metric=metric, d=i)
        d1, d0 = distortion(metric, embedding)
        if len(d) == 0:
            d.append(d0)
        d.append(d1)
    plt.title('Isomap embeddings with Pearson correlation distance')
    plt.xlabel('dimensions')
    plt.ylabel('distortion')
    plt.plot(range(9),d)
    plt.show()
    
    d = []
    for i in range(1, 9):
        metric = distance.euclidean
        embedding = mds(metric=metric, d=i)
        d1, d0 = distortion(metric, embedding)
        if len(d) == 0:
            d.append(d0)
        d.append(d1)
    plt.title('MDS embeddings with Euclidean distance')
    plt.xlabel('dimensions')
    plt.ylabel('distortion')
    plt.plot(range(9),d)
    plt.show()
    
    d = []
    for i in range(1, 9):
        metric = distance.correlation_distance
        embedding = mds(metric=metric, d=i)
        d1, d0 = distortion(metric, embedding)
        if len(d) == 0:
            d.append(d0)
        d.append(d1)
    plt.title('MDS embeddings with Pearson correlation distance')
    plt.xlabel('dimensions')
    plt.ylabel('distortion')
    plt.plot(range(9),d)
    plt.show()
    
    d = []
    for i in range(1, 9):
        metric = distance.euclidean
        embedding = laplacian(metric=metric, d=i)
        d1, d0 = distortion(metric, embedding)
        if len(d) == 0:
            d.append(d0)
        d.append(d1)
    plt.title('Laplacian eigenmap embeddings with Euclidean distance')
    plt.xlabel('dimensions')
    plt.ylabel('distortion')
    plt.plot(range(9),d)
    plt.show()
    
    d = []
    for i in range(1, 9):
        metric = distance.correlation_distance
        embedding = laplacian(metric=metric, d=i)
        d1, d0 = distortion(metric, embedding)
        if len(d) == 0:
            d.append(d0)
        d.append(d1)
    plt.title('Laplacian eigenmap embeddings with Pearson correlation distance')
    plt.xlabel('dimensions')
    plt.ylabel('distortion')
    plt.plot(range(9),d)
    plt.show()
    
    d = []
    for i in range(1, 9):
        metric = distance.euclidean
        embedding = lle(metric=metric, d=i)
        d1, d0 = distortion(metric, embedding)
        if len(d) == 0:
            d.append(d0)
        d.append(d1)
    plt.title('Locally linear embeddings')
    plt.xlabel('dimensions')
    plt.ylabel('distortion')
    plt.plot(range(9),d)
    plt.show()
 
    """
    method = sys.argv[1]
    
    if method == "isomap":
        f = isomap
    elif method == "mds":
        f = mds
    elif method == "lle":
        f = lle
    elif method == "laplacian":
        f = laplacian
    
    metric = sys.argv[2]
    
    if metric == "euclidean":
        df = distance.euclidean
    elif metric == "dynamic_time_warping":
        df = distance.dynamic_time_warping
    elif metric == "correlation":
        df = distance.correlation_distance
    
    dim = int(sys.argv[3])
    
    embeddings = f(metric=df, d=dim)
    #print(embeddings)
    print(distortion(df, embeddings))
    """

