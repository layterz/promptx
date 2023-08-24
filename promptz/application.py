import os
import importlib
import glob
import threading
import logging
from functools import partial
from typing import List
import nbformat
from nbconvert import HTMLExporter
from pydantic import create_model

from .world import World, System
from .api import API
from .admin import Admin
from .template import Template
from .models import openai


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
    
    @classmethod
    def from_config(cls, config, **kwargs):
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

        default_llm = config.get('DEFAULT_LLM')
        llm_str = default_llm.split('ai://')[-1]
        org, model, version = llm_str.split(':')
        LLM = get_llm(org, model)

        parsed_config = {
            'llm': LLM(version=version),
        }

        return cls(
            name='local',
            **{**parsed_config, **{k: v for k, v in kwargs.items() if v is not None}}
        )
    
    def _load(self, dir, cls):
        r = {}
        for file in glob.glob(os.path.join(dir, '*.py')):
            file_name = os.path.splitext(os.path.basename(file))[0]
            module = importlib.import_module(f'{dir}.{file_name}')
            for name, obj in vars(module).items():
                if isinstance(obj, type) and issubclass(obj, cls) and obj != cls:
                    r[name] = obj()
        return r
    
    def _load_templates(self):
        r = self._load(self.templates_dir, Template)
        t = r.values()
        print('t', t)
        return t
        for name, o in r.items():
            input = o.input.schema() if o.input else None
            print('o', o.output)
            
            if getattr(o.output, '_name', None) == 'List':
                inner = o.output.get('items')
                schema = inner.schema()
                output = {
                    'type': 'array',
                    'items': schema,
                    'definitions': schema.get('definitions', {})
                }
            else:
                output = o.output.schema() if o.output else None

        return t
    
    def _load_systems(self):
        r = self._load(self.systems_dir, System)
        s = []
        for name, o in r.items():
            s.append(dict({**dict(o)}))
        return s
    
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

