# ✅ 5 Casos de Regresión Lineal Simple

Se recomienda revisar la [📘 Guía Completa de Regresión Lineal para Machine Learning](GUIA.md)

## Tabla Comparativa General

| Aspecto | Caso 1 | Caso 2 | Caso 3 | Caso 4 | Caso 5 |
|---------|--------|--------|--------|--------|--------|
| **Tamaño Muestra** | 10 | 20 | 30 | 40 | 50 |
| **Correlación (r)** | 0.9986 | 0.9998 | 0.9978 | 0.9999 | 0.9860 |
| **R² Entrenamiento** | 0.9972 | 0.9999 | 0.9953 | 0.9997 | 0.9746 |
| **R² Prueba** | 0.9962 | 0.9991 | 0.9960 | 0.9996 | 0.9582 |
| **RMSE** | 1,501.21 | 5.37 | 3.19 | 2.80 | 2.78 |
| **MAE** | 1,500.00 | 3.91 | 2.60 | 2.58 | 2.40 |
| **Pendiente (β₁)** | 6,982.76 | 14.73 | 11.50 | 2.01 | 2.21 |
| **Intersección (β₀)** | 26,594.83 | 52.10 | -79.99 | 5.81 | 56.84 |

## 1. Interpretación de Resultados por Caso

### Caso 1: Experiencia vs Salario (10 registros)
**Contexto:** Recursos Humanos - Compensación Laboral

**Interpretación:**
- **Relación:** Experiencia laboral determina casi completamente el salario
- **Ecuación:** `Salario = 6,982.76 × Experiencia + 26,594.83`
- **Impacto:** Cada año de experiencia → +$6,982.76 en salario
- **Base:** Profesional sin experiencia gana $26,594.83
- **R² 99.72%:** Prácticamente toda la variación salarial se explica por experiencia

**Aplicación Práctica:**
- Estructura salarial basada en años de experiencia
- Presupuesto de compensaciones predecible
- Políticas de retención y desarrollo de carrera

**Documentación Completa del Análisis:** [Análisis y código](01.use_case.md)

### Caso 2: Publicidad vs Ventas (20 registros)
**Contexto:** Marketing - ROI de Campañas Publicitarias

**Interpretación:**
- **Relación:** Inversión publicitaria genera incremento casi perfecto en ventas
- **Ecuación:** `Ventas = 14.73 × Publicidad + 52.10`
- **Impacto:** Cada $1,000 en publicidad → +$14,730 en ventas
- **Base:** Ventas orgánicas de $52,100 sin publicidad
- **R² 99.91%:** Publicidad explica prácticamente todas las ventas

**Aplicación Práctica:**
- Optimización de presupuesto publicitario
- Proyecciones de ventas basadas en inversión
- Cálculo de ROI (1,372% de retorno)

**Documentación Completa del Análisis:** [Análisis y código](02.use_case.md)

### Caso 3: Temperatura vs Helados (30 registros)
**Contexto:** Negocios de Temporada - Gestión de Inventario

**Interpretación:**
- **Relación:** Temperatura determina demanda de helados
- **Ecuación:** `Ventas = 11.50 × Temperatura - 79.99`
- **Impacto:** Cada 1°C → +11.5 unidades de helados
- **Umbral:** A 7°C se alcanza punto de equilibrio (0 ventas)
- **R² 99.60%:** Clima explica casi toda la variación en ventas

**Aplicación Práctica:**
- Planificación de inventario según pronóstico climático
- Programación de personal en días calurosos
- Estrategias de precios adaptadas al clima

**Documentación Completa del Análisis:** [Análisis y código](03.use_case.md)

### Caso 4: Tamaño vs Precio Casa (40 registros)
**Contexto:** Bienes Raíces - Tasación Inmobiliaria

**Interpretación:**
- **Relación:** Metros cuadrados determinan prácticamente todo el precio
- **Ecuación:** `Precio = 2.01 × Metros² + 5.81`
- **Impacto:** Cada m² → +$2,012.5 en valor
- **Base:** Valor mínimo de $5,806 (costos administrativos)
- **R² 99.96%:** Tamaño explica casi todo el precio

**Aplicación Práctica:**
- Tasación automática de propiedades
- Valuación de proyectos inmobiliarios
- Decisiones de inversión y desarrollo

**Documentación Completa del Análisis:** [Análisis y código](04.use_case.md)

### Caso 5: Estudio vs Calificación (50 registros)
**Contexto:** Educación - Rendimiento Académico

**Interpretación:**
- **Relación:** Horas de estudio predicen rendimiento académico
- **Ecuación:** `Calificación = 2.21 × Horas + 56.84`
- **Impacto:** Cada hora de estudio → +2.21 puntos
- **Base:** 56.84 puntos sin estudio (5.7/10)
- **R² 95.82%:** Estudio explica la mayoría del rendimiento

**Aplicación Práctica:**
- Orientación académica personalizada
- Identificación temprana de estudiantes en riesgo
- Políticas de hábitos de estudio

**Documentación Completa del Análisis:** [Análisis y código](05.use_case.md)

## 2. Métricas de Rendimiento Comparativas

### Análisis de R² (Capacidad Explicativa)

| Caso | R² Entrenamiento | R² Prueba | Diferencia | Interpretación |
|------|------------------|-----------|------------|----------------|
| 1: Experiencia-Salario | 0.9972 | 0.9962 | 0.0010 | Excelente generalización |
| 2: Publicidad-Ventas | 0.9999 | 0.9991 | 0.0008 | Prácticamente perfecto |
| 3: Temperatura-Helados | 0.9953 | 0.9960 | -0.0007 | Mejora en prueba |
| 4: Tamaño-Precio | 0.9997 | 0.9996 | 0.0001 | Casi perfecto |
| 5: Estudio-Calificación | 0.9746 | 0.9582 | 0.0164 | Buen balance |

**Conclusiones sobre R²:**
- **Mejor Desempeño:** Caso 4 (Tamaño-Precio) con 0.9996
- **Más Realista:** Caso 5 (Estudio-Calificación) con 0.9582
- **Consistencia:** Todos los casos tienen R² > 0.95 en prueba
- **Generalización:** Diferencia mínima entre entrenamiento y prueba

### Análisis de Errores (RMSE vs MAE)

| Caso | RMSE | MAE | Diferencia | Interpretación |
|------|------|-----|------------|----------------|
| 1: Experiencia-Salario | $1,501 | $1,500 | $1 | Errores consistentes |
| 2: Publicidad-Ventas | $5,365 | $3,908 | $1,457 | Algunos errores moderados |
| 3: Temperatura-Helados | 3.19 | 2.60 | 0.59 | Errores controlados |
| 4: Tamaño-Precio | $2,801 | $2,575 | $226 | Muy consistentes |
| 5: Estudio-Calificación | 2.78 | 2.40 | 0.38 | Muy precisos |

**Conclusiones sobre Errores:**
- **Mayor Precisión:** Caso 5 (2.78 puntos de error)
- **Mayor Error Absoluto:** Caso 2 (mayor escala de valores)
- **Consistencia:** Relación RMSE/MAE cercana a 1 en todos los casos
- **Confiabilidad:** Todos los modelos son prácticamente precisos

### Comparación de Coeficientes

| Caso | Pendiente (β₁) | Intersección (β₀) | Interpretación |
|------|---------------|-------------------|----------------|
| 1 | 6,982.76 | 26,594.83 | Alto valor por año de experiencia |
| 2 | 14.73 | 52.10 | Alto ROI publicitario |
| 3 | 11.50 | -79.99 | Umbral de temperatura crítico |
| 4 | 2.01 | 5.81 | Valor consistente por m² |
| 5 | 2.21 | 56.84 | Puntaje base sin estudio |

**Conclusiones sobre Coeficientes:**
- **Mayor Pendiente:** Caso 1 (mayor escala de valores)
- **Pendiente Significativa:** Todos los casos tienen pendientes positivas significativas
- **Interpretabilidad:** Coeficientes tienen sentido en cada contexto


## 3. Ejecución de la Aplicación

### Instalación y Configuración

#### Requisitos del Sistema
```bash
# Python 3.8+
# Bibliotecas necesarias
pip install pandas numpy scikit-learn matplotlib seaborn
```

#### Estructura del Proyecto
```
IA-Python-Learning/
├── 00.datasets
│   ├── 01.salario_experiencia.csv
│   ├── 02.ventas_publicidad.csv
│   ├── 03.helados_temperatura.csv
│   ├── 04.precios_casas.csv
│   ├── 05.estudios_rendimiento.csv
│   └── note.txt
├── 01.regresion-lineal
│   ├── 01.salario_experiencia.py
│   ├── 02.ventas_publicidad.py
│   ├── 03.helados_temperatura.py
│   ├── 04.precios_casas.py
│   ├── 05.estudios_rendimiento.py
│   └── util_regresion_lineal.py
├── 01.use_case.md
├── 02.use_case.md
├── 03.use_case.md
├── 04.use_case.md
├── 05.use_case.md
└── README.md
```

```bash
python 01.regresion-lineal/[archivo según caso de uso].py
```