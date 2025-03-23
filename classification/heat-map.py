import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

# Cargar el dataset Iris en un DataFrame
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Calcular la matriz de correlación
corr_matrix = df.corr()

# Generar el heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

# Configuración del título
plt.title("Mapa de Calor de Correlación - Dataset Iris")
plt.show()

