import os
from loguru import logger
from rich import pretty
from rich.logging import RichHandler
import openai

from .world import World
from .api import API
from .admin import Admin
from .collection import ChromaVectorDB
from .models.openai import ChatGPT


class App:
    name: str
    path: str
    world: World

    def __init__(self, name, path, world=None, db=None, llm=None, **kwargs):
        self.name = name
        self.path = path
        self.world = world or World(name, db=db, default_llm=llm)
    
    @classmethod
    def load(cls, path, db=None, llm=None, env=None):
        db = db or ChromaVectorDB(path=path)

        if llm is None:
            default_llm_id = os.environ.get('PXX_DEFAULT_LLM', 'default')

            if default_llm_id == 'chatgpt':
                api_key = os.environ.get('PXX_OPENAI_API_KEY')
                org_id = os.environ.get('PXX_OPENAI_ORG_ID')
                openai.api_key = api_key or os.environ.get('OPENAI_API_KEY')
                openai.organization = org_id or os.environ.get('OPENAI_ORG_ID')
                llm = llm or ChatGPT(id='default', api_key=env.get('PXX_OPENAI_API_KEY'), org_id=env.get('PXX_OPENAI_ORG_ID'))

        config = {
            'name': 'local',
            'path': path,
            'db': db,
            'llm': llm,
        }

        pretty.install()

        env = {**os.environ, **(env or {})}
        log_file_path = f"./log/{env.get('PXX_ENV', 'development')}.log"
        level = env.get('PXX_LOG_LEVEL', 'INFO')

        # Configure Loguru
        logger.remove()
        logger.add(
            log_file_path,
            rotation="10 MB",
            level=level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            backtrace=True
        )

        logger.info("Log file: " + log_file_path)
        logger.info("Log level: " + level)

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
    
    def __repr__(self):
        return f'<App {self.name} path={self.path}>'

