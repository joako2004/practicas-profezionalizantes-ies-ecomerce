from flask import Flask, render_template, request, jsonify
from chat import get_response

app = Flask(__name__)

# Ruta para servir la página principal con el chat
@app.get("/")
def index_get():
    return render_template('base.html')  # carga la plantilla base.html

# Ruta que recibe las peticiones AJAX desde el front y devuelve la respuesta del bot
@app.post("/predict")
def predict():
    # obtiene el JSON y extrae el campo "message"
    text = request.get_json().get("message")

    # validación: revisar que text exista, sea cadena y no esté vacío
    if not isinstance(text, str) or not text.strip():
        # si no es válido, devolvemos un error 400 con mensaje apropiado
        return jsonify({"answer": "Por favor envía un mensaje válido."}), 400

    # si pasó la validación, llamamos a la función que genera la respuesta
    response = get_response(text)

    # empaquetamos la respuesta en formato JSON
    message = {"answer": response}
    return jsonify(message)

# arranca la aplicación en modo debug para desarrollo
if __name__ == "__main__":
    app.run(debug=True)
