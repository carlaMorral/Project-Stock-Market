import sys
import pandas as pd
import numpy as np

from collections import defaultdict
from distance import *
from plot import *
from single_linkage import *
from spectral import *


def standarize_data(data):
    """
    Computes Amplitude Scaling of the given time series
    """
    # compute the row-wise mean and std
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)

    # subtract the mean and divide by the std
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
print(data.shape)

# Single-linkage clustering
eps = 500
metric = euclidean
data_norm = standarize_data(data)
labels = single_linkage_clustering(data, metric, eps)
print("Labels:")
print(labels)
clusters = defaultdict(list)
for i, label in enumerate(labels):
    value = index_name = series.index[i]
    clusters[label].append(value)
with open("sl_euclid.txt", "w") as sl:
    for label, comps in clusters.items():
        sl.write("Cluster: " + str(label) + ", companies: " + str(comps) + "\n")

# TODO probar mes metriques (Dynamic Time Warping?) pero es n**2, hauriem de reduir les dades


# TODO spectral clustering
# eps = 0.2
# ggraph = get_input_graph(data, eps=eps, type="gaussian")
# predg = spectral_clustering(k=2, graph=ggraph)
# show(data, predg, "Spectral clustering - Gaussian graph prediction, eps=" + str(eps))

