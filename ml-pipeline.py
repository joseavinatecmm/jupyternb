# Importar librerías necesarias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Cargar el dataset
# Vamos a crear un dataset sintético para este ejemplo
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Convertir a un DataFrame de pandas para simular un dataset real
data = pd.DataFrame(np.hstack([X, y]), columns=['Feature', 'Target'])
print("Dataset original:")
print(data.head())

# 2. Preprocesamiento del dataset
# Verificar valores faltantes
print("\nValores faltantes:")
print(data.isnull().sum())

# No hay valores faltantes en este ejemplo, pero si los hubiera, podríamos manejarlos así:
# data = data.fillna(data.mean())  # Rellenar con la media

# 3. Ingeniería de características
# En este caso, solo tenemos una característica, pero podríamos agregar más si fuera necesario.
# Por ejemplo, podríamos agregar una columna con el cuadrado de la característica:
data['Feature_squared'] = data['Feature'] ** 2

# Separar características (X) y variable objetivo (y)
X = data[['Feature']]  # Usamos solo la característica original para la regresión lineal simple
y = data['Target']

# 4. Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Escalar características (opcional para regresión lineal, pero buena práctica)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 7. Hacer predicciones
y_pred = model.predict(X_test_scaled)

# 8. Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nEvaluación del modelo:")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Coeficiente de Determinación (R^2): {r2:.2f}")

# 9. Visualizar los resultados
plt.scatter(X_test, y_test, color='blue', label='Datos reales')
plt.plot(X_test, y_pred, color='red', label='Predicciones')
plt.xlabel('Característica')
plt.ylabel('Target')
plt.title('Regresión Lineal Simple')
plt.legend()
plt.show()

# 10. Optimización (opcional)
# En este caso, la regresión lineal no tiene hiperparámetros que optimizar, pero podríamos usar técnicas como GridSearchCV
# para modelos más complejos.

# 11. Guardar el modelo (opcional)
import joblib
joblib.dump(model, 'linear_regression_model.pkl')

# 12. Cargar el modelo (opcional)
# loaded_model = joblib.load('linear_regression_model.pkl')
