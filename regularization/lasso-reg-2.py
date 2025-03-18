# Importar librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generar un dataset sintético con un COD del 95%
np.random.seed(42)
n_samples = 100
X = np.linspace(0, 10, n_samples)

# Crear una relación lineal con un poco de ruido
true_coef = 2.5  # Coeficiente verdadero
true_intercept = 1.0  # Intercepto verdadero
y_true = true_intercept + true_coef * X  # Relación lineal verdadera

# Añadir ruido controlado para lograr un COD del 95%
desired_r2 = 0.95
noise_variance = np.var(y_true) * (1 / desired_r2 - 1)
y = y_true + np.random.normal(0, np.sqrt(noise_variance), n_samples)

# Calcular el COD del dataset original
original_r2 = r2_score(y_true, y)
print(f"Coeficiente de Determinación (COD) del dataset original: {original_r2:.4f}")

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

# Calcular el COD en entrenamiento y prueba
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nCOD en entrenamiento (Regresión Polinomial): {train_r2:.4f}")
print(f"COD en prueba (Regresión Polinomial): {test_r2:.4f}")

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

# Calcular el COD en entrenamiento y prueba con Lasso
train_r2_lasso = r2_score(y_train, y_train_pred_lasso)
test_r2_lasso = r2_score(y_test, y_test_pred_lasso)

print(f"\nCOD en entrenamiento (Lasso Regression): {train_r2_lasso:.4f}")
print(f"COD en prueba (Lasso Regression): {test_r2_lasso:.4f}")

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
