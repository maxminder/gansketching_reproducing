import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
newX = pca.fit_transform(X)
newX /= np.linalg.norm(newX, axis=-1, keepdims=True)
print(X)
print(newX)
print(pca.explained_variance_ratio_)