import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def analizar_regresion(df, x_col, y_col, titulo, test_size=0.2):
    """
    Función para realizar análisis completo de regresión lineal simple
    """
    print(f"\n{'='*50}")
    print(f"CASO DE USO: {titulo}")
    print(f"{'='*50}")
    
    # 1. Información básica del dataset
    print(f"\n1. INFORMACIÓN DEL DATASET:")
    print(f"   - Registros: {len(df)}")
    print(f"   - Columnas: {df.columns.tolist()}")
    print(f"   - Rango de {x_col}: {df[x_col].min()} a {df[x_col].max()}")
    print(f"   - Rango de {y_col}: {df[y_col].min():.2f} a {df[y_col].max():.2f}")
    
    # 2. Correlación entre variables
    correlacion = df[x_col].corr(df[y_col])
    print(f"\n2. CORRELACIÓN:")
    print(f"   - Coeficiente de correlación: {correlacion:.4f}")
    
    # 3. Preparación de datos
    X = df[[x_col]]
    y = df[y_col]
    
    # 4. División en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    print(f"\n3. DIVISIÓN DE DATOS:")
    print(f"   - Entrenamiento: {len(X_train)} registros")
    print(f"   - Prueba: {len(X_test)} registros")
    
    # 5. Entrenamiento del modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 6. Coeficientes del modelo
    print(f"\n4. MODELO ENTREÑADO:")
    print(f"   - Pendiente (coeficiente): {model.coef_[0]:.4f}")
    print(f"   - Intersección: {model.intercept_:.4f}")
    print(f"   - Ecuación: {y_col} = {model.coef_[0]:.4f} * {x_col} + {model.intercept_:.4f}")
    
    # 7. Predicciones
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # 8. Evaluación del modelo
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    print(f"\n5. EVALUACIÓN DEL MODELO:")
    print(f"   - R² (entrenamiento): {r2_train:.4f}")
    print(f"   - R² (prueba): {r2_test:.4f}")
    print(f"   - RMSE (prueba): {rmse_test:.4f}")
    print(f"   - MAE (prueba): {mae_test:.4f}")
    
    # 9. Visualización
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Análisis de Regresión: {titulo}', fontsize=14)
    
    # Gráfico 1: Regresión con datos de entrenamiento
    axes[0].scatter(X_train, y_train, alpha=0.7, label='Entrenamiento')
    axes[0].plot(X_train, y_pred_train, 'r-', label='Línea de regresión')
    axes[0].set_xlabel(x_col)
    axes[0].set_ylabel(y_col)
    axes[0].set_title('Regresión - Entrenamiento')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Gráfico 2: Regresión con datos de prueba
    axes[1].scatter(X_test, y_test, alpha=0.7, label='Prueba')
    axes[1].plot(X_test, y_pred_test, 'r-', label='Predicciones')
    axes[1].set_xlabel(x_col)
    axes[1].set_ylabel(y_col)
    axes[1].set_title('Regresión - Prueba')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Gráfico 3: Predicciones vs Valores Reales
    axes[2].scatter(y_test, y_pred_test, alpha=0.7)
    axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[2].set_xlabel('Valores Reales')
    axes[2].set_ylabel('Predicciones')
    axes[2].set_title('Predicciones vs Reales')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 10. Predicción de ejemplo
    print(f"\n6. EJEMPLO DE PREDICCIÓN:")
    ejemplo_x = X.mean().values[0]
    ejemplo_y_pred = model.predict([[ejemplo_x]])[0]
    print(f"   - Para {x_col} = {ejemplo_x:.2f}")
    print(f"   - Predicción de {y_col} = {ejemplo_y_pred:.2f}")
    
    return model