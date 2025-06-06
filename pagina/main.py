from flask import Blueprint, render_template

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
