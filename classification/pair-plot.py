import seaborn as sns
import matplotlib.pyplot as plt

# Cargar dataset de ejemplo
df = sns.load_dataset("iris")

# Crear Pairplot
sns.pairplot(df, hue="species", diag_kind="kde")

# Mostrar gráfico
plt.show()
