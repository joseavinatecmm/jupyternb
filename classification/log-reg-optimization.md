# Regresión Logística y Optimización con Gradiente Descendente

**Author:** J. Antonio Aviña Méndez

La **regresión logística** es un modelo de clasificación binaria que estima la probabilidad de que una observación pertenezca a una de dos clases (\( y \in \{0,1\} \)). Se basa en la función **sigmoide** para modelar la relación entre las características de entrada y la probabilidad de una clase específica.

$$
h_\theta(x) = \sigma(\theta^T x)
$$

$$
h_\theta(x) = \sigma(w^T x)
$$


donde:
- $h_\theta(x)$ es la probabilidad de que $y = 1$ dado $x$.
- $\sigma(z)$ es la función sigmoide:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

- $\theta$ es el vector de parámetros a optimizar.
- $x$ es el vector de características (i.e. features). 

## 1. Función de Pérdida: Entropy Loss (Cross-Entropy)

El entrenamiento del modelo de regresión logística se basa en minimizar la **función de pérdida de entropía cruzada (Cross-Entropy Loss)**, también conocida como **Log Loss**. Su expresión matemática es:

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]
$$

(i.e. The Log likelihood)

donde:
- $m$ es el número de muestras en el conjunto de datos.
- $y^{(i)}$ es la etiqueta real de la muestra $i$.
- $h_\theta(x^{(i)})$ es la probabilidad predicha para la muestra $i$.

La **intuición** detrás de esta función de pérdida es que:
- Si $y^{(i)} = 1$, la función penaliza fuertemente si $h_\theta(x^{(i)})$ es cercano a 0.
- Si  $y^{(i)} = 0$, la función penaliza fuertemente si $h_\theta(x^{(i)})$ es cercano a 1.

## 2. Gradiente de la Función de Pérdida

Para optimizar $\theta$, calculamos el **gradiente de $J(\theta)$ con respecto a \( \theta \)**:

$$\prod\limits_{i=1}^N \sigma(\mathbf{w}^T \mathbf{x}_i)^{y_i} \cdot (1 - \sigma(\mathbf{w}^T \mathbf{x}_i))^{1 - y_i}.
$$



$$
\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}
$$



Este gradiente indica cuánto debemos modificar $\theta_j$ para minimizar la función de pérdida.

## 3. Algoritmo de Gradiente Descendente

El **gradiente descendente** actualiza iterativamente los parámetros $\theta$ según la siguiente regla:

$$
\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}
$$

donde:
- $\alpha$ es la **tasa de aprendizaje**.
- $\frac{\partial J(\theta)}{\partial \theta_j}$ es la derivada de la función de pérdida con respecto a $\theta_j$.

## 4. Implementación Paso a Paso

1. **Inicializar $\theta$** con valores pequeños (ej. ceros o valores aleatorios pequeños).
2. **Repetir hasta convergencia**:
   - Calcular las predicciones $h_\theta(x)$.
   - Evaluar la función de pérdida $J(\theta)$.
   - Calcular el gradiente $\frac{\partial J(\theta)}{\partial \theta_j}$.
   - Actualizar $\theta$ con la regla de gradiente descendente.
3. **Detener** cuando los cambios en $\theta$ sean suficientemente pequeños o después de un número máximo de iteraciones.

## 5. Interpretación Geométrica

El gradiente descendente ajusta los parámetros $\theta$ moviéndose en la dirección de máxima disminución de $J(\theta)$, buscando minimizar la diferencia entre las predicciones $h_\theta(x)$ y las etiquetas reales $y$.

## 6. Regularización

Para evitar sobreajuste, se puede agregar un término de regularización $\lambda$ en la función de pérdida:

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2
$$

En este caso, el gradiente modificado es:

$$
\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} \theta_j
$$

Este término de regularización $L2$ (Ridge) penaliza valores grandes en $\theta_j$, reduciendo la complejidad del modelo y mejorando la generalización.

## 7. Conclusión

El entrenamiento de la regresión logística se basa en minimizar la **función de entropía cruzada (Cross-Entropy Loss)** utilizando **gradiente descendente**, ajustando los parámetros $\theta$ para mejorar las predicciones del modelo.

