from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Generar datos de juguete
X, y = make_classification(
    n_samples=200,      # 200 “frutitas” a etiquetar
    n_features=5,       # 5 características por fruta
    n_informative=3,    # 3 de esas 5 son útiles
    random_state=0      # para que siempre sean las mismas frutas
)

# 2. Separar “frutitas” en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,      # 30% de las frutas son de prueba
    random_state=0
)

# 3. Crear el “árbol mágico” (bosque de 20 árboles)
clf = RandomForestClassifier(
    n_estimators=20,    # número de árboles
    random_state=0
)
clf.fit(X_train, y_train)   # el bosque aprende de las frutas de entrenamiento

# 4. El bosque etiqueta las frutas de prueba
y_pred = clf.predict(X_test)

# 5. Mostrar reporte de cómo lo hizo
print(classification_report(y_test, y_pred))

# ¿Qué hace cada línea?
# make_classification: “prepara una canasta de frutas” con etiquetas falsas (0 o 1).
# train_test_split: “separa la canasta” en dos: una de práctica (entrenamiento) y otra para evaluar (prueba).
# RandomForestClassifier: imagina 20 árboles que votan si cada fruta es “roja” (1) o “no roja” (0).
# fit: los árboles aprenden mirando ejemplos buenos.
# predict: los árboles votan en las frutas nuevas.
# classification_report:
# Precision: de todas las “manzanas rojas” que dijo el modelo, ¿cuántas lo eran de verdad?
# Recall: de todas las manzanas rojas que había, ¿cuántas encontró?
# F1-score: un número que mezcla precision y recall para ver un balance.