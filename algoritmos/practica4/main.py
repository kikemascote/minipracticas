import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# 1. Cargar Titanic
df = sns.load_dataset('titanic')[['sex', 'age', 'fare', 'survived']].dropna()

# 2. Convertir “sex” a número
df['sex'] = LabelEncoder().fit_transform(df['sex'])

# 3. Entrenar y probar
X_train, X_test, y_train, y_test = train_test_split(df[['sex', 'age', 'fare']], df['survived'], test_size=0.3)
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

# 4. Probabilidades y AUC
probs = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)
print(f"AUC-ROC: {auc:.2f}")

# LabelEncoder: cambia “male/female” a 0/1.
# predict_proba: te dice “qué tan seguro está el modelo de que sobrevivas”.
# AUC-ROC: número entre 0.5 y 1; más cercano a 1 → mejor predicción.
