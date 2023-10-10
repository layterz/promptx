import os
import sys
from typing import Callable, List

from .collection import Collection 
from .world import Session
from .application import App
from .auth import DefaultUser 


def prompt(instructions=None, input=None, output=None, prompt=None, context=None, template=None, llm=None, examples=None, num_examples=1, history=None, tools=None, dryrun=False, retries=3, debug=False, silent=False, **kwargs):
    kwargs = dict(
        instructions=instructions,
        input=input,
        output=output,
        prompt=prompt,
        llm=llm,
        context=context,
        examples=examples,
        num_examples=num_examples,
        history=history,
        tools=tools,
        template=template,
        debug=debug,
        dryrun=dryrun,
        silent=silent,
        retries=retries,
        store=store,
    )
    return DEFAULT_SESSION.prompt(**kwargs)


def chain(*steps, **kwargs):
    return DEFAULT_SESSION.chain(*steps, **kwargs)


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


def chat(message, input=None, **kwargs):
    return DEFAULT_SESSION.chat(message, input, **kwargs)


def evaluate(*testcases, **kwargs) -> Collection:
    return DEFAULT_SESSION.evaluate(*testcases, **kwargs)


def history(query=None, **kwargs):
    if query is not None:
        return DEFAULT_SESSION.history(query, **kwargs)
    else:
        return DEFAULT_SESSION.history


def templates(ids=None, **kwargs) -> Collection:
    return DEFAULT_SESSION.world.templates(ids=ids, **kwargs)


def systems(ids=None, **kwargs) -> Collection:
    return DEFAULT_SESSION.world.systems(ids=ids, **kwargs)


def session() -> Session:
    return DEFAULT_SESSION


def run(world=None, **kwargs):
    if world is None:
        world = DEFAULT_WORLD
    return world(**kwargs)


DEFAULT_WORLD = None
def set_default_world(world):
    global DEFAULT_WORLD
    DEFAULT_WORLD = world


DEFAULT_SESSION = None
def set_default_session(session):
    global DEFAULT_SESSION
    DEFAULT_SESSION = session


Embedding = List[float]
EmbedFunction = Callable[[List[str]], List[Embedding]]


def find_project_root(path=None, config_filename='.pz.env'):
    home_dir = os.path.expanduser("~")
    if path is None:
        path = os.getcwd()
    while path != home_dir:
        if os.path.exists(os.path.join(path, config_filename)):
            return path
        path = os.path.dirname(path)
    return None


def load_local_config(path=None, filename=".pz.env"):
    home_dir = os.path.expanduser("~")
    current_dir = os.getcwd()
    path = path or current_dir

    while current_dir != home_dir:
        file_path = os.path.join(current_dir, filename)

        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                config = {
                    line.split('=', 1)[0].strip(): line.split('=', 1)[1].strip()
                    for line in f if '=' in line
                }
                return (DefaultUser(), config)

        current_dir = os.path.dirname(current_dir)

    file_path = os.path.join(home_dir, filename)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            config = {
                line.split('=', 1)[0].strip(): line.split('=', 1)[1].strip()
                for line in f if '=' in line
            }
            return (DefaultUser(), config)

    return None, None


def load(path='local', llm=None, ef=None, logger=None, **kwargs):
    if path == 'local':
        path = find_project_root()
    
    if path is None:
        raise ValueError('could not find project')

    app = None
    if path.startswith('http'):
        print('loading remote app from', path)
        raise NotImplementedError
    else:
        sys.path.append(path)
        user, config = load_local_config(path)
        app = App.from_config(path, config, llm=llm, ef=ef, logger=logger, **kwargs)
    
    print(f'Loaded {app}')
    
    s = app.world.create_session(user)
    set_default_session(s)
    return app