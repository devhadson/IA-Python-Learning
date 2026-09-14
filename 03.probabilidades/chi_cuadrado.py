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