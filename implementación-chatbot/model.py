import torch                    # framework principal de deep learning
import torch.nn as nn           # módulo de construcciones de redes neuronales

# Definición de la clase de la red neuronal
class NeuralNet(nn.Module):
    # Constructor: define las capas y la función de activación
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()  
        # Capa lineal de entrada → oculta
        self.l1 = nn.Linear(input_size, hidden_size)
        # Segunda capa lineal oculta → oculta
        self.l2 = nn.Linear(hidden_size, hidden_size)
        # Capa lineal oculta → salida (una neurona por cada clase/tag)
        self.l3 = nn.Linear(hidden_size, num_classes)
        # Función de activación ReLU (Rectified Linear Unit)
        self.relu = nn.ReLU()
        
    # Método que define el paso hacia adelante (forward pass)
    def forward(self, x):
        # aplica la primera capa y luego ReLU
        out = self.l1(x)
        out = self.relu(out)
        
        # aplica la segunda capa y luego ReLU
        out = self.l2(out)
        out = self.relu(out)
        
        # aplica la capa de salida (sin activación, dejaremos que CrossEntropyLoss
        # internamente aplique softmax si es necesario)
        out = self.l3(out)
    
        return out                # devuelve logits sin normalizar
