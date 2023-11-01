import os
import time
from typing import Callable, List
from loguru import logger

from .collection import Collection, MemoryVectorDB
from .world import Session
from .application import App
from .auth import DefaultUser 


def prompt(instructions=None, input=None, output=None, prompt=None, context=None, template=None, llm=None, examples=None, allow_none=False, history=None, tools=None, dryrun=False, retries=3, debug=False, silent=False, **kwargs):
    kwargs = dict(
        instructions=instructions,
        input=input,
        output=output,
        prompt=prompt,
        llm=llm,
        context=context,
        examples=examples,
        allow_none=allow_none,
        history=history,
        tools=tools,
        template=template,
        debug=debug,
        dryrun=dryrun,
        silent=silent,
        retries=retries,
    )
    return DEFAULT_SESSION.prompt(**kwargs)


def store(*items, collection=None, **kwargs) -> Collection:
    return DEFAULT_SESSION.store(*items, collection=collection, **kwargs)


def delete(*items, collection=None, **kwargs) -> None:
    return DEFAULT_SESSION.delete(*items, collection=collection, **kwargs)


def collections():
    return DEFAULT_SESSION.collections()


def delete_collection(collection, **kwargs) -> None:
    return DEFAULT_SESSION.delete_collection(collection, **kwargs)


def create_collection(collection, **kwargs) -> None:
    return DEFAULT_SESSION.create_collection(collection, **kwargs)


def query(*texts, field=None, where=None, collection=None, **kwargs) -> Collection:
    return DEFAULT_SESSION.query(
        *texts, field=field, where=where, collection=collection, **kwargs)


def chat(message, context=None, **kwargs):
    return DEFAULT_SESSION.chat(message, context, **kwargs)


def evaluate(*testcases, **kwargs) -> Collection:
    return DEFAULT_SESSION.evaluate(*testcases, **kwargs)


def session() -> Session:
    return DEFAULT_SESSION


DEFAULT_SESSION = None
def set_default_session(session):
    global DEFAULT_SESSION
    DEFAULT_SESSION = session


def find_project_root(path=None, config_filename='.px'):
    home_dir = os.path.expanduser("~")
    if path is None:
        path = os.getcwd()
    while path != home_dir:
        if os.path.exists(os.path.join(path, config_filename)) and os.path.isdir(path):
            return path
        path = os.path.dirname(path)
    if os.path.exists(os.path.join(home_dir, config_filename)):
        return home_dir
    return None


def load(path='local', **env):
    if path == 'local':
        project_path = find_project_root()
        if project_path is not None:
            path = project_path

    app = None
    if path == 'local':
        db = MemoryVectorDB()
        app = App.load(path, db=db, env=env)
    elif path.startswith('http'):
        print('loading remote app from', path)
        raise NotImplementedError
    else:
        logger.info(f'loading local app from {path}')
        from dotenv import load_dotenv
        load_dotenv(os.path.join(path, '.env'), override=True)
        logger.info(f'loaded environment variables from {os.path.join(path, ".env")}')
        logger.info(f'API KEY {os.environ.get("PXX_OPENAI_API_KEY")[-5:]}')
        app = App.load(path, env=env)

        # look for a config.py file and execute it
        config_file = os.path.join(path, 'config.py')
        if os.path.exists(config_file):
            import importlib.util
            spec = importlib.util.spec_from_file_location('config', config_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            session = app.world.create_session(f'{app.name}_config_{int(time.time())}')
            module.setup(session)
    
    user = DefaultUser()
    s = app.world.create_session(user)
    set_default_session(s)
    return app