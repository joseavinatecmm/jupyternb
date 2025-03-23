### Contexto del Challenge 1:
En la regresión logística, queremos modelar la probabilidad de que una observación $\mathbf{x}_i$ pertenezca a la clase $1$ (en lugar de la clase $0$). El modelo se define como:

$
P(y_i = 1 | \mathbf{x}_i; \mathbf{w}) = \sigma(\mathbf{w}^T \mathbf{x}_i) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x}_i}},
$

donde:
- $\mathbf{x}_i$ es el vector de características de la observación  i-ésima.
- $\mathbf{w}$ es el vector de pesos (parámetros del modelo).
- $\sigma(z) = \frac{1}{1 + e^{-z}}$ es la función sigmoide.

Para un conjunto de datos $\mathcal{D} = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \dots, (\mathbf{x}_N, y_N)\}$, donde $y_i \in \{0, 1\}$, queremos encontrar los valores de $\mathbf{w}$ que maximizan la verosimilitud de los datos.

---

### Función de verosimilitud:
La probabilidad de observar una etiqueta $y_i$ dado $\mathbf{x}_i$ y los parámetros $\mathbf{w}$ es:

$
P(y_i | \mathbf{x}_i; \mathbf{w}) = \sigma(\mathbf{w}^T \mathbf{x}_i)^{y_i} \cdot (1 - \sigma(\mathbf{w}^T \mathbf{x}_i))^{1 - y_i}.
$

La función de verosimilitud conjunta para todos los datos es:

$
\mathcal{L}(\mathbf{w}) = \prod\limits_{i=1}^N P(y_i | \mathbf{x}_i; \mathbf{w}) =$ 

$\prod\limits_{i=1}^N \sigma(\mathbf{w}^T \mathbf{x}_i)^{y_i} \cdot (1 - \sigma(\mathbf{w}^T \mathbf{x}_i))^{1 - y_i}.
$

---

### Challenge 1
El objetivo es encontrar el vector de pesos $\mathbf{w}$ que maximiza la verosimilitud:

$
\mathbf{w}^* = \arg\max_{\mathbf{w}} \mathcal{L}(\mathbf{w}) = \arg\max_{\mathbf{w}} \prod\limits_{i=1}^N \left[\sigma(\mathbf{w}^T \mathbf{x}_i)^{y_i} \cdot (1 - \sigma(\mathbf{w}^T \mathbf{x}_i))^{(1 - y_i)} \ \right]
$

---
