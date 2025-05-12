from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

digits = load_digits()
X = digits.images.reshape(-1,8,8,1)/16.0
y = digits.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

model = Sequential([
    Flatten(input_shape=(8,8,1)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile('adam','sparse_categorical_crossentropy',metrics=['accuracy'])
hist = model.fit(X_train, y_train, epochs=5, verbose=0, validation_split=0.2)

# Mostrar evolución de accuracy
import matplotlib.pyplot as plt
plt.plot(hist.history['accuracy'], label='train')
plt.plot(hist.history['val_accuracy'], label='val')
plt.title("Accuracy durante entrenamiento")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

print("Test accuracy:", model.evaluate(X_test, y_test, verbose=0)[1])

# Flatten: aplana la imagen de 8×8 a 64 “dibujitos”.
# accuracy: porcentaje de dígitos bien adivinados.
# El gráfico te muestra cómo mejora el modelo en cada pasada (época).