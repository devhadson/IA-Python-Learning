# red_neuronal_basica.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. Preparación de datos simulados (Clasificación Binaria)
# Generamos 500 muestras con 2 características cada una
np.random.seed(42)
torch.manual_seed(42)

X_np = np.random.uniform(-1, 1, (500, 2))
Y_np = (X_np[:, 0]**2 + X_np[:, 1]**2 > 0.4).astype(np.float32)

X = torch.tensor(X_np, dtype=torch.float32)
Y = torch.tensor(Y_np, dtype=torch.float32).unsqueeze(1)

# 2. Definición de la Arquitectura de la Red Neuronal
class RedNeuronalBasica(nn.Module):
    def __init__(self):
        super(RedNeuronalBasica, self).__init__()
        # Capa oculta: recibe 2 entradas, tiene 8 neuronas
        self.capa_oculta = nn.Linear(2, 8)
        # Función de activación no lineal
        self.relu = nn.ReLU()
        # Capa de salida: recibe 8 entradas, produce 1 salida
        self.capa_salida = nn.Linear(8, 1)
        # Activación final para obtener probabilidad entre 0 y 1
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.capa_oculta(x)
        x = self.relu(x)
        x = self.capa_salida(x)
        x = self.sigmoid(x)
        return x

# 3. Inicialización del Modelo, Función de Pérdida y Optimizado
modelo = RedNeuronalBasica()
funcion_perdida = nn.BCELoss()  # Entropía cruzada binaria
optimizador = optim.SGD(modelo.parameters(), lr=0.1)  # Gradiente descendiente estocástico

# 4. Bucle de Entrenamiento
epochs = 300
print("Iniciando entrenamiento...")

for epoch in range(epochs):
    # Paso hacia adelante (Forward pass)
    predicciones = modelo(X)
    perdida = funcion_pérdida(predicciones, Y) if 'funcion_perdida' in locals() else funcion_perdida(predicciones, Y)
    
    # Paso hacia atrás (Backward pass) y optimización
    optimizador.zero_grad()  # Limpiar gradientes anteriores
    perdida.backward()       # Calcular gradientes (Backpropagation)
    optimizador.step()       # Actualizar pesos y sesgos
    
    # Mostrar progreso cada 50 épocas
    if (epoch + 1) % 50 == 0:
        print(f"Época [{epoch+1}/{epochs}] - Pérdida: {perdida.item():.4f}")

print("¡Entrenamiento finalizado!")

# 5. Evaluación rápida
con_exito = ((predicciones > 0.5) == Y).float().mean()
print(f"Precisión final del modelo: {con_exito.item() * 100:.2f}%")
