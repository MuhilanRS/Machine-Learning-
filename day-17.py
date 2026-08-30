import numpy as np
from sklearn.cluster import KMeans

X = np.array([
    [2, 3],
    [3, 4],
    [4, 3],
    [10, 11],
    [11, 10],
    [12, 11]
])

model = KMeans(n_clusters=2, random_state=42, n_init=10)

model.fit(X)

print(model.labels_)
print(model.cluster_centers_)