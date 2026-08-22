import numpy as np
import matplotlib.pyplot as plt

# 1. Configuración de aleatoriedad para reproducibilidad
np.random.seed(45)

# 2. Generación del dataset sintético (50 puntos de datos)
X = np.random.uniform(10, 50, size=50)

# Generamos Y usando la ecuación de la recta (y = 1.46x + 13.22) + ruido aleatorio
ruido = np.random.normal(0, 7.5, size=50)
Y = 1.46 * X + 13.22 + ruido

# 3. Puntos extremos para trazar la línea de regresión continua
X_linea = np.linspace(8, 51, 100)
Y_linea = 1.46 * X_linea + 13.22

# 4. Construcción y estilizado de la gráfica con Matplotlib
plt.figure(figsize=(8, 6))

# Dibujar los puntos dispersos (Datos observados)
plt.scatter(X, Y, color='#8cb3ef', edgecolor='#2c3e50', s=65, alpha=0.9, label='Datos observados', zorder=3)

# Dibujar la recta de regresión
plt.plot(X_linea, Y_linea, color='#e74c3c', linewidth=2.5, label='Línea de regresión: y = 1.46x + 13.22', zorder=4)

# Personalización de títulos y etiquetas de los ejes
plt.title('Regresión Lineal Simple', fontsize=13, pad=12, fontweight='normal')
plt.xlabel('Variable Independiente (X)', fontsize=11)
plt.ylabel('Variable Dependiente (Y)', fontsize=11)

# Configuración de límites y cuadrícula (Estilo idéntico a la imagen)
plt.xlim(6, 53)
plt.ylim(21, 101)
plt.grid(True, linestyle='--', color='#d1d5db', alpha=0.7, zorder=1)

# Estilo de la leyenda
plt.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=10)

# Limpieza estética del marco (remover líneas superior y derecha)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#4b5563')
ax.spines['bottom'].set_color('#4b5563')

# Mostrar el gráfico en pantalla
plt.tight_layout()
plt.show()