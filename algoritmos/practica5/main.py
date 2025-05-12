import pandas as pd
import itertools, math
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'id': ['D', 'A', 'B', 'C'],
    'x': [50, 20, 80, 20],
    'y': [50, 20, 80, 80]
})


def dist(i, j):
    xi, yi = df.loc[df.id == i, ['x', 'y']].values[0]
    xj, yj = df.loc[df.id == j, ['x', 'y']].values[0]
    return math.hypot(xi - xj, yi - yj)


# Calcular “ahorro” al unir rutas
pairs = list(itertools.combinations(['A', 'B', 'C'], 2))
savings = {f"{i}-{j}": dist('D', i) + dist('D', j) - dist(i, j) for i, j in pairs}
print("Savings:", savings)

# Gráfico de barras
plt.bar(savings.keys(), savings.values())
plt.title("Ahorro al unir dos clientes")
plt.xlabel("Par de clientes")
plt.ylabel("Distancia ahorrada")
plt.show()

# Savings: cuánto camino te ahorras si visitas A y B juntos en lugar de por separado.
# La barra más alta → mejor par para unir primero.
