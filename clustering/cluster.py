import sys
import pandas as pd
import numpy as np

from collections import defaultdict
import distance
from plot import *
from single_linkage import *


def standarize_data(data):
    """
    Computes Amplitude Scaling of the given time series
    """
    # compute the row-wise mean and std
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)

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

# Single-linkage clustering
def single_linkage(metric, eps=20):
    data_norm = standarize_data(data)
    labels = single_linkage_clustering(data_norm, metric, eps)
    return labels


if __name__=="__main__":
    method = sys.argv[1]
    metric = sys.argv[2]

    labels = locals()[method](getattr(distance, metric))
    outfile = method + "_" + metric + ".txt"

    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        value = index_name = series.index[i]
        clusters[label].append(value)
    
    with open(outfile, "w") as sl:
        for label, comps in clusters.items():
            sl.write("Cluster: " + str(label) + ", companies: " + str(comps) + "\n")

# TODO probar mes metriques (Dynamic Time Warping?) pero es n**2, hauriem de reduir les dades


# TODO spectral clustering


