import os
import importlib
import glob
import threading
import logging
from functools import partial

from .world import World
from .api import API
from .admin import Admin
from .template import Template
from .models import openai
from .collection import ChromaVectorDB


class App:
    name: str
    path: str
    world: World

    def __init__(self, name, path, world=None, db=None):
        self.name = name
        self.path = path
        self.world = world or World(name, db=db)
    
    @classmethod
    def load(cls, path):
        db = ChromaVectorDB(path=os.path.join(path, 'db'))

        config = {
            'name': 'local',
            'path': path,
            'db': db,
        }

        return cls(**config)

    @property
    def api(self):
        if self._api is None:
            self._api = API(self.world)
        return self._api

    @property
    def admin(self):
        if self._admin is None:
            self._admin = Admin(self.world)
        return self._admin

    def _load(self, dir, cls):
        r = {}
        for file in glob.glob(os.path.join(dir, '*.py')):
            file_name = os.path.splitext(os.path.basename(file))[0]
            package_name = dir.split('/')[-1]
            module = importlib.import_module(f'{package_name}.{file_name}')
            for name, obj in vars(module).items():
                if isinstance(obj, type) and issubclass(obj, cls) and obj != cls:
                    r[name] = obj(id=name)
        return r
    
    def _load_templates(self, templates_dir=None):
        if templates_dir is None:
            return []
        ts = self._load(templates_dir, Template)
        return ts.values()
    
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
    
    def __repr__(self):
        return f'<App {self.name} path={self.path}>'

