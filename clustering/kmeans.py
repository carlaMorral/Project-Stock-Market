import numpy as np

from sklearn.metrics import pairwise_distances_argmin


def kmeans(X, k, tol=1e-4):
    # Initialize centers
    idx = np.random.randint(X.shape[0], size=k)
    centers = X[idx,:]

    converged = False
    while not converged:
        # Assign points to closest center
        labels = pairwise_distances_argmin(X, centers)

        # Find new centers
        new_centers = np.array([X[labels == i].mean(0) for i in range(k)])

        # Check for convergence
        if np.linalg.norm(centers - new_centers) <= tol:
            converged = True
        else: centers = new_centers
    
    return labels
