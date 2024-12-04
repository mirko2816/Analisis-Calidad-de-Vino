import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Leer el archivo CSV
df = pd.read_csv('WineQT.csv')
# Mostrar las primeras filas del archivo
df.head()

# Transformar la variable respuesta 'quality' en una variable binaria: "bueno" y "no bueno"
df['quality_bin'] = df['quality'].apply(lambda x: 'bueno' if x >= 7 else 'no bueno')

# Crear la figura con dos subgráficos
fig, axes = plt.subplots(2, 1, figsize=(6, 8))  # 2 filas y 1 columna, tamaño 8x12

# Gráfico 1: Histograma para la distribución de 'quality'
sns.histplot(df['quality'], bins=range(df['quality'].min(), df['quality'].max() + 1), kde=False, color='skyblue', edgecolor='black', ax=axes[0])
axes[0].set_title('Histograma de la Calidad del Vino')
axes[0].set_xlabel('Calidad del Vino')
axes[0].set_ylabel('Cantidad')
axes[0].set_xticks(range(df['quality'].min(), df['quality'].max() + 1))

# Agregar las cantidades sobre cada barra del histograma
for p in axes[0].patches:
    axes[0].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height() + 10), ha='center', va='bottom')

# Gráfico 2: Distribución de la variable respuesta con cantidades
sns.countplot(x='quality_bin', data=df, ax=axes[1], hue='quality_bin', palette='Set2', legend=False)

# Agregar las cantidades sobre cada barra del gráfico de barras
for p in axes[1].patches:
    axes[1].annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height() + 10),
                     ha='center', va='bottom', fontsize=12)

axes[1].set_title('Distribución de la Calidad del Vino (Bueno vs No Bueno)')
axes[1].set_xlabel('Calidad del Vino')
axes[1].set_ylabel('Cantidad')

# Ajustar el espacio entre los subgráficos
plt.subplots_adjust(hspace=0.3)  # Ajustamos el espacio vertical entre los gráficos

# Mostrar los gráficos
plt.show()