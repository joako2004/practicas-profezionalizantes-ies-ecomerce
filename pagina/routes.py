from flask import Blueprint, render_template

# blueprint “main”
main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/noticias')
def noticias():
    return render_template('noticias.html')

@main.route('/inicio-sesion')
def inicio_sesion():
    return render_template('inicio-sesion.html')

@main.route('/registro')
def registro():
    return render_template('registro.html')

# blueprint “comics”
comics = Blueprint('comics', __name__, url_prefix='/comics')

@comics.route('/')
def lista_comics():
    return render_template('comics.html')
