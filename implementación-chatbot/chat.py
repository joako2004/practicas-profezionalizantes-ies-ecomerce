import random                               # para seleccionar respuestas aleatorias
import json                                 # para cargar el archivo de intents (patrones y respuestas)
import torch                                # PyTorch para manejar el modelo entrenado
from model import NeuralNet                 # clase de la red neuronal definida en model.py
from nltk_utils import bag_of_words, tokenize  # funciones para procesar texto

# elegimos GPU si está disponible, sino CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# cargamos los intents desde el JSON para consultar respuestas
with open("intents.json", "r") as file:
    intents = json.load(file)
    
# cargamos los datos guardados tras el entrenamiento
FILE = "data.pth"
data = torch.load(FILE)

# extraemos dimensiones y vocabulario
input_size = data["input_size"]     # tamaño de la entrada (largo del bag-of-words)
hidden_size = data["hidden_size"]   # tamaño de la capa oculta
output_size = data["output_size"]   # número de etiquetas (clases)
all_words = data["all_words"]       # vocabulario (lista de palabras)
tags = data["tags"]                 # lista de etiquetas
model_state = data["model_state"]   # pesos entrenados del modelo

# reconstruimos el modelo y cargamos sus pesos
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()  # modo evaluación (desactiva dropout, gradientes, etc.)

bot_name = 'SAM'  # nombre que usará el bot al responder

def get_response(msg):
    """
    Dado un mensaje de texto msg (string), devuelve la respuesta generada por el modelo.
    """
    # 1) tokenizar la oración de entrada
    sentence = tokenize(msg)
    # 2) convertir tokens a vector "bag of words"
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])            # ajustar dimensiones para el modelo (batch_size=1)
    X = torch.from_numpy(X).to(device)      # convertir a tensor de PyTorch

    # 3) obtener la salida del modelo
    output = model(X)
    # 4) elegir la etiqueta con mayor puntuación
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # 5) calcular probabilidades (softmax) y extraer la de la etiqueta predicha
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # 6) si la confianza es alta, buscar la etiqueta en intents y devolver respuesta aleatoria
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    # 7) si la confianza es baja, devolver mensaje por defecto
    return "I do not understand..."

# permite probar el bot desde consola
if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":  # permite salir escribiendo "quit"
            break
        resp = get_response(sentence)   # obtener respuesta del bot
        print(f"{bot_name}: {resp}")    # mostrar nombre del bot y su respuesta
