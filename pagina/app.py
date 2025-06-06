from flask import Flask
from routes import main, comics

def create_app():
    app = Flask(
        __name__,
        template_folder='templates',
        static_folder='static'
    )

    # registrar blueprints
    app.register_blueprint(main)
    app.register_blueprint(comics)

    return app

if __name__ == '__main__':
    create_app().run(debug=True)
