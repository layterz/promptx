import os
from loguru import logger
from rich import pretty
from rich.logging import RichHandler

from .world import World
from .api import API
from .admin import Admin
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
    def load(cls, path, db=None, env=None):
        db = db or ChromaVectorDB(path=path)

        config = {
            'name': 'local',
            'path': path,
            'db': db,
        }

        pretty.install()

        env = env or os.environ
        log_file_path = f"./log/{env.get('PX_ENV', 'development')}.log"  # Replace 'development' with your environment
        level = env.get('PX_LOG_LEVEL', 'INFO')  # Replace 'CRITICAL' with your log level

        # Configure Loguru
        logger.remove()
        logger.add(
            log_file_path,
            rotation="10 MB",  # Rotate log files that exceed 10 MB
            level=level,  # Minimum level to log
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            backtrace=True  # Enable traceback
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

