import os
import importlib
import glob
import threading
import logging
import json
import nbformat
from nbconvert import HTMLExporter

from .world import World, System
from .api import API
from .admin import Admin
from .template import Template


class App:
    name: str
    world: World
    templates_dir: str = 'templates'
    systems_dir: str = 'systems'
    notebooks_dirs: str = ['notebooks']

    def __init__(self, name, world=None, llm=None, ef=None, logger=None, db=None):
        self.name = name
        self.logger = logger or logging.getLogger(self.name)
        templates = self._load_templates()
        systems = self._load_systems()
        notebooks = self._load_notebooks()
        self.world = world or World(name, templates=templates, systems=systems, notebooks=notebooks, llm=llm, ef=ef, logger=logger, db=db)
        self.api = API(self.world)
        self.admin = Admin(self.world)
    
    def _load(self, dir, cls):
        r = {}
        for file in glob.glob(os.path.join(dir, '*.py')):
            file_name = os.path.splitext(os.path.basename(file))[0]
            module = importlib.import_module(f'{dir}.{file_name}')
            for name, obj in vars(module).items():
                if isinstance(obj, type) and issubclass(obj, cls) and obj != cls:
                    r[name] = dict(obj())
        return r
    
    def _load_templates(self):
        return self._load(self.templates_dir, Template)
    
    def _load_systems(self):
        return self._load(self.systems_dir, System)
    
    def _load_notebooks(self):
        html_notebooks = {}
        for dir in self.notebooks_dirs:
            for file in glob.glob(os.path.join(dir, '*.ipynb')):
                with open(file, 'r') as f:
                    file_name = os.path.splitext(os.path.basename(file))[0]
                    html_exporter = HTMLExporter()
                    notebook_node = nbformat.read(f, as_version=4)
                    body, resources = html_exporter.from_notebook_node(notebook_node)
                    html_notebooks[file_name] = body
        return html_notebooks
    
    def _serve_api(self, host='0.0.0.0', port=8000):
        import uvicorn
        uvicorn.run(self.api.fastapi_app, host=host, port=port)
    
    def _serve_admin(self, host='0.0.0.0', port=8001):
        from waitress import serve
        serve(self.admin.app.server, host=host, port=port)
    
    def serve(self, host='0.0.0.0', port=8000, admin_port=8001):
        api = threading.Thread(target=lambda: self._serve_api(host=host, port=port))
        admin = threading.Thread(target=lambda: self._serve_admin(host=host, port=admin_port))
        api.start()
        admin.start()

