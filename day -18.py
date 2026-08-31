from sklearn.cluster import KMeans

X = [[1, 1], [2, 2], [8, 8], [9, 9]]

model = KMeans(n_clusters=2, random_state=42)

model.fit(X)

print(model.labels_)
print(model.cluster_centers_)