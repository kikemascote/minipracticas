from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

iris = load_iris()
X2 = PCA(n_components=2).fit_transform(iris.data)
labels = KMeans(n_clusters=3, random_state=0).fit_predict(X2)

# Plot en 2D
plt.scatter(X2[:,0], X2[:,1], c=labels)
plt.title("Clusters en 2D tras PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# PCA: “voltear” los datos para verlos en dos ejes que expliquen más información.
# c=labels: colorea cada punto según el grupo al que pertenece.