from flask import Flask

def create_app():
    app = Flask(__name__, 
                template_folder='../template',
                static_folder='../statics')
    
    # Register blueprints
    from .routes import main
    app.register_blueprint(main)
    
    return app 