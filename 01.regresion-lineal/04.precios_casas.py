import pandas as pd
from util_regresion_lineal import analizar_regresion

def main():
    """
    Función principal que ejecuta todos los casos de uso
    """
    print("ANÁLISIS DE 5 CASOS DE USO DE REGRESIÓN LINEAL SIMPLE")
    print("="*70)
        
    # Caso 4: Tamaño de Casa vs Precio
    df4 = pd.read_csv('00.datasets/04.precios_casas.csv')
    modelo4 = analizar_regresion(
        df4, 'Metros_Cuadrados', 'Precio', 
        'Relación Tamaño-Precio Casa (40 registros)'
    )
    
    print("\n" + "="*70)
    print("CONCLUSIONES GENERALES:")
    print("="*70)
    print("1. Los 5 casos de uso demuestran la versatilidad de la regresión lineal simple")
    print("2. La calidad del modelo depende de la correlación entre variables")
    print("3. Con más registros, el modelo tiende a ser más estable")
    print("4. Los resultados deben interpretarse en contexto del dominio específico")
    print("5. Es importante validar supuestos de linealidad y normalidad de residuos")

if __name__ == "__main__":
    main()