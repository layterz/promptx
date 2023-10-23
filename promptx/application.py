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

    def __init__(self, name, path, world=None, llm=None, ef=None, logger=None, db=None, templates_dir=None):
        self.name = name
        self.path = path
        self.logger = logger or logging.getLogger(self.name)
        templates = self._load_templates(templates_dir)
        self.world = world or World(name, templates=templates, llm=llm, ef=ef, logger=logger, db=db)
        self.api = API(self.world)
        self.admin = Admin(self.world)
    
    @classmethod
    def load(cls, path):
        db = ChromaVectorDB(path=os.path.join(path, 'db'))

        config = {
            'name': 'local',
            'path': path,
            'db': db,
        }

        return cls(**config)

    @classmethod
    def from_config(cls, path, config, **kwargs):
        def get_llm(org, model):
            if org == 'openai':
                auth = {
                    'api_key': config.get('OPENAI_API_KEY'),
                    'org_id': config.get('OPENAI_ORG_ID'),
                }
                if model == 'chatgpt':
                    return partial(openai.ChatGPT, **auth)
                elif model == 'instructgpt':
                    return partial(openai.InstructGPT, **auth)
            
            raise Exception(f'Unknown LLM config: {org}:{cls}')

        default_llm = config.get('DEFAULT_LLM', 'ai://openai:chatgpt:latest')
        templates_dir = os.path.join(path, config.get('TEMPLATES_DIR', 'templates'))
        llm_str = default_llm.split('ai://')[-1]
        org, model, version = llm_str.split(':')
        LLM = get_llm(org, model)
        db = ChromaVectorDB(path=path)

        parsed_config = {
            'name': config.get('NAME', 'local'),
            'path': path,
            'llm': LLM(version=version),
            'db': db,
            'templates_dir': templates_dir,
        }

        return cls(
            **{**parsed_config, **{k: v for k, v in kwargs.items() if v is not None}}
        )
    
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

