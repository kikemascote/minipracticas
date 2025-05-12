import numpy as np
from sklearn.cluster import DBSCAN

cluster = np.random.randn(50,2)*0.3 + np.array([2,2])
outliers = np.array([[0,0],[4,4],[0,4],[4,0]])
data = np.vstack([cluster, outliers])

db = DBSCAN(eps=0.5, min_samples=5).fit(data)
indices = np.where(db.labels_==-1)[0]
print("Índices de outliers:", indices)

# DBSCAN: dibuja círculos de radio eps y si un punto no tiene suficientes vecinos, lo marca como “fuera del grupo” (outlier).
# Los números que veas (p.ej. [50 51 52 53]) son las posiciones de esos puntos raros.