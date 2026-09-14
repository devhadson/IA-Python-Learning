Prueba de chi-cuadrado en Python utilizando la función `chi2_contingency` de la librería SciPy para tablas de contingencia o independencia.

## Código de Ejemplo

Puedes usar este código básico para evaluar la asociación entre dos variables categóricas:

```python
import numpy as np
import scipy.stats as stats

# Crear la tabla de frecuencias observadas
observado = np.array([[50, 30], [20, 40]])

# Ejecutar el test de chi-cuadrado
chi2_stat, p_val, dof, expected = stats.chi2_contingency(observado)

print("Tabla de frecuencias observadas:")
print(observado)

print(f"Estadístico Chi-cuadrado: {chi2_stat:.4f}")
print(f"Valor p: {p_val:.4f}")
print(f"Grados de libertad: {dof}")
print("Frecuencias esperadas:")
print(expected)
```

> Ejecuta el codigo desde el archivo [`chi_cuadrado.py`](chi_cuadrado.py) con el comando: `python chi_cuadrado.py`

## Interpretación Básica
- **Estadístico Chi-cuadrado:** Mide la discrepancia entre los valores observados y los esperados.
- **Valor p (`p_val`):** Si es menor que tu nivel de significancia (por lo general 0.05), rechazas la hipótesis nula y concluyes que existe una relación significativa entre las variables