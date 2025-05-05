import nltk                       # biblioteca para procesamiento de lenguaje natural
import re                         # expresiones regulares para manipulación de texto
import unicodedata                # para normalización de caracteres Unicode (acentos)
import numpy as np                # biblioteca para cálculo numérico (vectores, matrices)
from nltk.stem.porter import PorterStemmer  # algoritmo de stemming

# Creamos el stemmer (algoritmo que reduce palabras a su raíz)
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    Recibe una frase (string) y devuelve una lista de tokens (palabras y signos).
    Usa el tokenizador de NLTK.
    Ejemplo: "¡Hola, mundo!" → ["¡", "Hola", ",", "mundo", "!"]
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    Limpia y aplica stemming a una palabra:
      1. Elimina signos de puntuación al inicio y al final.
      2. Normaliza caracteres Unicode para quitar acentos.
      3. Convierte a minúsculas.
      4. Aplica el algoritmo PorterStemmer.
    Devuelve la raíz (stem) de la palabra.
    """
    # 1) Quitar signos de puntuación del principio o final
    word = re.sub(
        r'^[¿¡\?!\.,;:\'"()]+|[¿¡\?!\.,;:\'"()]+$',
        '',
        word
    )
    # 2) Descomponer caracteres Unicode y eliminar marcas de acento
    nfkd_form = unicodedata.normalize('NFKD', word)
    word = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    # 3) Pasar todo a minúsculas
    word = word.lower()
    # 4) Devolver la versión stemizada
    return stemmer.stem(word)

def bag_of_words(tokenized_sentence, all_words):
    """
    Dada una oración tokenizada y el vocabulario all_words,
    devuelve un vector (numpy array) de la misma longitud que all_words,
    con 1.0 en cada posición cuya palabra aparece en la oración,
    y 0.0 en las que no.
    
    Pasos:
      a) Stemizar cada token de la oración.
      b) Crear un vector de ceros.
      c) Recorrer el vocabulario stemizado y marcar con 1.0 si está en la oración.
    """
    # a) Stemizar cada palabra/token de la oración
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    # b) Inicializar el "bolsillo de palabras" con ceros
    bag = np.zeros(len(all_words), dtype=np.float32)

    # Stemizar también el vocabulario para que la comparación sea consistente
    all_words = [stem(w) for w in all_words]

    # c) Marcar con 1.0 las posiciones cuyas palabras aparecen en la oración
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag
