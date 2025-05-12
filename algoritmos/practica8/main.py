import numpy as np
from scipy.sparse.linalg import svds

R = np.array([[5,3,0],[4,0,0],[1,1,0],[0,0,5],[0,0,4]])
u,s,vt = svds(R, k=2)
print("Dimensiones de U, Σ, Vᵀ:", u.shape, s.shape, vt.shape)
# svds: descompone la matriz en tres para “comprenderla” con menos números.
# Verás algo como (5,2), (2,), (2,6); indica la forma de cada trozo.