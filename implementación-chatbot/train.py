import json                          # para cargar el archivo JSON de “intents” (patrones y respuestas)
import torch                         # PyTorch, framework de deep learning
import numpy as np                   # para convertir listas a arreglos numéricos
import torch.nn as nn                # componentes de redes neuronales
from torch.utils.data import Dataset, DataLoader  # para crear dataset y loader
from nltk_utils import tokenize, stem, bag_of_words  # funciones de preprocesado de texto
from model import NeuralNet          # definición de la arquitectura de la red

# -------------------------
# 1) Carga y preparativos
# -------------------------
with open("intents.json", "r") as file:
    intents = json.load(file)        # carga el JSON de intenciones

all_words = []    # lista que contendrá todas las palabras del vocabulario
tags = []         # lista de etiquetas (clases)
xy = []           # lista de tuplas (tokens, etiqueta) para cada patrón

# Recorre cada intención para extraer patrones y etiquetas
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)                # guarda la etiqueta
    
    # para cada frase de ejemplo (pattern) en esa etiqueta
    for pattern in intent['patterns']:
        w = tokenize(pattern)       # tokeniza la frase en palabras
        all_words.extend(w)         # añade esos tokens al vocabulario temporal
        xy.append((w, tag))         # guarda la pareja (tokens, etiqueta)

# palabras de puntuación que queremos ignorar
ignore_words = ['¿', '¡', '?', '!', '.', ',']

# aplica stemming y limpieza, y filtra las palabras vacías
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))   # elimina duplicados y ordena

tags = sorted(set(tags))             # elimina duplicados y ordena etiquetas

# -------------------------
# 2) Creación de datos de entrenamiento
# -------------------------
x_train = []    # lista de vectores “bag of words”
y_train = []    # lista de etiquetas (índices)

# convierte cada par (tokens, tag) en un vector numérico y su etiqueta entera
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)  # vector de ceros y unos
    x_train.append(bag)
    
    label = tags.index(tag)    # índice de la etiqueta en la lista tags
    y_train.append(label)

# transforma a numpy arrays para PyTorch
x_train = np.array(x_train)
y_train = np.array(y_train)

# -------------------------
# 3) Definición del Dataset
# -------------------------
class ChatDataset(Dataset):
    def __init__(self):
        # número de muestras
        self.n_samples = len(x_train)
        # datos de entrada y salida
        self.x_data = x_train
        self.y_data = y_train
        
    def __getitem__(self, index):
        # devuelve la muestra e etiqueta en la posición index
        return self.x_data[index], self.y_data[index]
        
    def __len__(self):
        # devuelve el total de muestras
        return self.n_samples

# -------------------------
# 4) Hiperparámetros
# -------------------------
batch_size = 8
hidden_size = 8
output_size = len(tags)           # tantas neuronas de salida como etiquetas
input_size = len(x_train[0])      # tamaño del vector bag_of_words
learning_rate = 0.001
num_epochs = 1000

# crea el dataset y el loader para iterar en batches
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2)

# selecciona dispositivo (GPU si está disponible)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------
# 5) Construcción del modelo
# -------------------------
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()            # función de pérdida para clasificación
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# -------------------------
# 6) Bucle de entrenamiento
# -------------------------
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)    # mueve datos a GPU/CPU
        labels = labels.to(device)
        
        outputs = model(words)      # forward pass
        loss = criterion(outputs, labels)  # calcula pérdida
        
        optimizer.zero_grad()       # limpia gradientes previos
        loss.backward()             # backpropagation
        optimizer.step()            # actualiza pesos
        
    # cada 100 epochs, imprime el progreso
    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

# imprime la pérdida final
print(f'Final loss: loss={loss.item():.4f}')

# -------------------------
# 7) Guardar el modelo entrenado
# -------------------------
data = {
    'model_state': model.state_dict(),  # pesos de la red
    'input_size': input_size,
    'output_size': output_size,
    'hidden_size': hidden_size,
    'all_words': all_words,            # vocabulario
    'tags': tags                       # lista de etiquetas
}

FILE = 'data.pth'
torch.save(data, FILE)                 # serializa en disco

print(f'training complete. file saved to {FILE}')
