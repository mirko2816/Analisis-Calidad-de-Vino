import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# Leer el archivo CSV
df = pd.read_csv('WineQT.csv')

# Transformar la variable respuesta 'quality' en una variable binaria: "bueno" y "no bueno"
df['quality_bin'] = df['quality'].apply(lambda x: 'bueno' if x >= 7 else 'no bueno')

# Separar las clases mayoritaria y minoritaria
df_majority = df[df['quality_bin'] == 'no bueno']
df_minority = df[df['quality_bin'] == 'bueno']

# Aumentar la clase minoritaria (upsampling)
df_minority_upsampled = resample(df_minority,
                                  replace=True,     # muestreo con reemplazo
                                  n_samples=len(df_majority),    # para que coincida con la clase mayoritaria
                                  random_state=42)  # resultados reproducibles

# Combinar clases
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Imprimir la distribución de clases antes y después del upsampling
print("Distribución original:")
for clase, conteo in df['quality_bin'].value_counts().items():
    print(f"{clase}: {conteo}")

print("\nDistribución después de upsampling:")
for clase, conteo in df_upsampled['quality_bin'].value_counts().items():
    print(f"{clase}: {conteo}")

# Separar las variables explicativas (X) y la variable respuesta (y)
X = df_upsampled.drop(columns=['quality', 'quality_bin'])
y = df_upsampled['quality_bin']

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos balanceados en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Inicializar el modelo de Regresión Logística
logreg = LogisticRegression(random_state=42, max_iter=500)

# Entrenar el modelo
logreg.fit(X_train, y_train)

# Predecir con el conjunto de prueba
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Imprimir los resultados
print(f"\nPrecisión del modelo: {accuracy:.4f}")
print("\nMatriz de confusión:")
print(conf_matrix)
print("\nReporte de clasificación:")
print(class_report)