import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Crear señal de ruido + onda (como un “bocadillo” de vibración)
t = np.arange(0, 20, 0.1)
serie = np.sin(t) + 0.1 * np.random.randn(len(t))

# 2. Función para armar “ventanas” de 10 puntos y su siguiente valor
def create_seq(data, w=10):
    X, y = [], []
    for i in range(len(data)-w):
        X.append(data[i:i+w])
        y.append(data[i+w])
    return np.array(X), np.array(y)

X, y = create_seq(serie)
X = X.reshape((X.shape[0], X.shape[1], 1))
split = int(len(X)*0.7)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 3. Definir la red LSTM
model = Sequential([
    LSTM(5, input_shape=(X.shape[1],1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)

# 4. Predecir y calcular error medio
preds = model.predict(X_test)
mse = np.mean((preds.flatten() - y_test)**2)
print(f"MSE (error medio cuadrático): {mse:.4f}")

# 5. Graficar: línea real vs línea predicha
plt.plot(y_test, label='Real')
plt.plot(preds, label='Predicho')
plt.legend()
plt.title("Real vs Predicho")
plt.xlabel("Índice de la muestra")
plt.ylabel("Valor")
plt.show()

# np.sin + ruido: dibuja una ola y le pone “picazón” al azar.
# ventanas: es como mirar 10 fotos de la ola y adivinar la siguiente.
# LSTM: una pequeña memoria que ve esas 10 fotos y trata de adivinar la siguiente.
# MSE: número que dice “qué tan lejos estuvo la predicción” (cero sería perfecto).
# plt.show(): abre la ventana con el dibujo de la ola real y la ola que predijo.