import numpy as np
import networkx as nx
from distance import *


def single_linkage_clustering(data, metric, epsilon):
    graph = nx.Graph()
    for i in np.arange(data.shape[0]):
        for j in np.arange(i, data.shape[0]):
            if metric(data[i], data[j]) < epsilon:
                graph.add_edge(i,j)

    labels = np.zeros(data.shape[0])
    cc = nx.connected_components(graph)
    for i, component in enumerate(cc):
        for node in component:
            labels[node] = i

    return labels
