# Importar librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error

# 1. Generar un dataset sintético no lineal
np.random.seed(42)
X = np.linspace(0, 10, 100)  # 100 puntos entre 0 y 10
y = np.sin(X) + np.random.normal(0, 0.2, 100)  # y = sin(X) + ruido

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Aplicar regresión polinomial de alto grado (para provocar overfitting)
# Crear características polinomiales (grado 15)
poly = PolynomialFeatures(degree=15, include_bias=False)
X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
X_test_poly = poly.transform(X_test.reshape(-1, 1))

# Entrenar un modelo de regresión lineal con características polinomiales
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predecir en los conjuntos de entrenamiento y prueba
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

# Calcular el error (MSE) en entrenamiento y prueba
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"MSE en entrenamiento (Regresión Polinomial): {train_mse:.4f}")
print(f"MSE en prueba (Regresión Polinomial): {test_mse:.4f}")

# 3. Visualizar el overfitting
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Datos de entrenamiento')
plt.scatter(X_test, y_test, color='green', label='Datos de prueba')
plt.plot(np.sort(X), model.predict(poly.transform(np.sort(X).reshape(-1, 1))), color='red', label='Regresión Polinomial (Overfit)')
plt.title("Overfitting en Regresión Polinomial (Grado 15)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# 4. Aplicar Lasso Regression para evitar overfitting
# Lasso aplica regularización L1, que penaliza los coeficientes grandes
lasso_model = Lasso(alpha=0.1)  # alpha es el parámetro de regularización
lasso_model.fit(X_train_poly, y_train)

# Predecir en los conjuntos de entrenamiento y prueba
y_train_pred_lasso = lasso_model.predict(X_train_poly)
y_test_pred_lasso = lasso_model.predict(X_test_poly)

# Calcular el error (MSE) en entrenamiento y prueba
train_mse_lasso = mean_squared_error(y_train, y_train_pred_lasso)
test_mse_lasso = mean_squared_error(y_test, y_test_pred_lasso)

print(f"\nMSE en entrenamiento (Lasso Regression): {train_mse_lasso:.4f}")
print(f"MSE en prueba (Lasso Regression): {test_mse_lasso:.4f}")

# 5. Visualizar el resultado con Lasso Regression
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Datos de entrenamiento')
plt.scatter(X_test, y_test, color='green', label='Datos de prueba')
plt.plot(np.sort(X), lasso_model.predict(poly.transform(np.sort(X).reshape(-1, 1))), color='orange', label='Lasso Regression')
plt.title("Lasso Regression para Evitar Overfitting")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
