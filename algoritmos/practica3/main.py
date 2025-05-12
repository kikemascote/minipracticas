import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Mini-corpus
df = pd.DataFrame({
    'text': ['Bueno', 'Malo', 'Excelente', 'Pésimo', 'Satisfactorio', 'Deficiente'],
    'label': [1, 0, 1, 0, 1, 0]
})

# 1. Transformer de texto a números
vec = TfidfVectorizer()
X = vec.fit_transform(df.text)

# 2. Máquina de vectores de soporte
clf = LinearSVC()
clf.fit(X, df.label)

# 3. Probar con nuevas frases
tests = vec.transform(['Muy bueno', 'No sirve', 'Super excelente', 'Extra bueno'])
print("Predicciones:", clf.predict(tests))

# TfidfVectorizer: cuenta palabras y las pondera según su importancia (“palabra rara → peso alto”).
# LinearSVC: aprende una línea que separa “palabras buenas” de “malas”.
# predict: te da un array de 1 (positivo) o 0 (negativo).
