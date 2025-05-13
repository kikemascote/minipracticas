Mini-Prácticas de Algoritmos de Análisis de Datos
Este repositorio contiene nueve mini-prácticas didácticas que ilustran flujos básicos de Machine Learning, Deep Learning, NLP, series de tiempo, clustering, recomendación y detección de anomalías. Cada práctica es un script autónomo en Python que puede ejecutarse tal cual en PyCharm, VS Code o desde línea de comandos.

📂 Estructura del repositorio
bash
Copiar
Editar
MiniPracticas/
├── practica1/    # Random Forest (clasificación binaria)
│   └── main.py
├── practica2/    # LSTM (serie de tiempo sintética)
│   └── main.py
├── practica3/    # TF-IDF + SVM (mini-corpus de texto)
│   └── main.py
├── practica4/    # Regresión logística (Titanic)
│   └── main.py
├── practica5/    # Heurística Savings (VRP simplificado)
│   └── main.py
├── practica6/    # Red neuronal sencilla para dígitos (Digits)
│   └── main.py
├── practica7/    # PCA + k-Means (Iris)
│   └── main.py
├── practica8/    # SVD para recomendación (matriz de ratings)
│   └── main.py
└── practica9/    # IsolationForest (detección de anomalías IoT)
    └── main.py
⚙️ Requisitos Previos
Python 3.8+

pip (gestor de paquetes)

Entorno virtual (altamente recomendado)

Instalación de dependencias
bash
Copiar
Editar
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow
Nota:

En Windows con Python 3.13 puede no encontrarse tensorflow.

En ese caso usa pip install tensorflow-cpu o baja a Python 3.10.x.

▶️ Cómo ejecutar cada práctica
Abre tu IDE (PyCharm, VS Code, etc.) y carga este proyecto.

Activa tu entorno virtual.

Navega a la carpeta de la práctica que quieras probar:

bash
Copiar
Editar
cd practicaX
Ejecuta el script:

bash
Copiar
Editar
python main.py
Observa en consola las métricas y, si aplica, se abrirá una ventana con el gráfico.
