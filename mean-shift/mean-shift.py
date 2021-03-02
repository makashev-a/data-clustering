import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from sklearn.cluster import MeanShift

style.use("ggplot")

from sklearn.datasets._samples_generator import make_blobs

centers = [[2, 2], [4, 5], [3, 10]]
X, _ = make_blobs(n_samples=500, centers=centers, cluster_std=1)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

ms = MeanShift()
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

print(cluster_centers)
n_clusters_ = len(np.unique(labels))
print("Estimated clusters:", n_clusters_)
