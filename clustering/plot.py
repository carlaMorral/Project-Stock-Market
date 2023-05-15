from sklearn.datasets import make_circles, make_moons, make_blobs
from matplotlib import pyplot
from pandas import DataFrame

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def show(X, Y, title="default", num_colors=100, show_colors=True):
    X = np.transpose(X)
    if show_colors:
        sns.jointplot(x=X[0], y=X[1], kind="scatter", space=0, hue=Y, palette=sns.color_palette(None, num_colors), legend=False)
    else:
        sns.jointplot(x=X[0], y=X[1], kind="scatter", space=0)
    plt.suptitle(title)
    plt.show()

def show_3d(X, title="default", labels=None):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x = X[:,0]
    y = X[:,1]
    z = X[:,2]
    cmap = ListedColormap(sns.color_palette(None,10).as_hex())
    if labels is not None:
        sc = ax.scatter3D(x, y, z, s=30, c=labels, cmap=cmap)
        plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
    else:
        ax.scatter3D(x, y, z)
    plt.suptitle(title)
    plt.show()

