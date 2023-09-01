import os
import sys
from typing import Callable, List

from .collection import Collection 
from .world import World, Session
from .application import App


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


def query(*texts, field=None, where=None, collection=None, **kwargs) -> Collection:
    return DEFAULT_SESSION.query(
        *texts, field=field, where=where, collection=collection, **kwargs)


def chat(input, **kwargs):
    return {}


def evaluate(*testcases, **kwargs) -> Collection:
    return DEFAULT_SESSION.evaluate(*testcases, **kwargs)


def history(query=None, **kwargs):
    if query is not None:
        return DEFAULT_SESSION.history(query, **kwargs)
    else:
        return DEFAULT_SESSION.history


def templates(ids=None, **kwargs) -> Collection:
    return DEFAULT_SESSION.world.templates(ids=ids, **kwargs)


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


def load_app(path, config=None, **kwargs):
    import os
    import sys
    
    module_path = os.path.abspath(path)

    if module_path not in sys.path:
        sys.path.append(module_path)
    
    os.environ['PROMPTZ_PATH'] = path
    try:
        from app import create_app
        return create_app()
    except ModuleNotFoundError:
        return App.from_config(config, **kwargs)


def load_config(filename=".pz.env"):
    home_dir = os.path.expanduser("~")
    current_dir = os.getcwd()

    while current_dir != home_dir:
        file_path = os.path.join(current_dir, filename)

        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                config = {
                    line.split('=', 1)[0].strip(): line.split('=', 1)[1].strip()
                    for line in f if '=' in line
                }
                return (current_dir, config)

        current_dir = os.path.dirname(current_dir)

    file_path = os.path.join(home_dir, filename)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            config = {
                line.split('=', 1)[0].strip(): line.split('=', 1)[1].strip()
                for line in f if '=' in line
            }
            return (home_dir, config)

    return None, None


def load(llm=None, ef=None, logger=None, log_format='notebook', **kwargs):
    path, config = load_config()
    sys.path.append(path)
    if path is None:
        w = World(
            'local', llm=llm, ef=ef, logger=logger, 
            templates=[], systems=[], notebooks={}, **kwargs)
        s = w.create_session(log_format=log_format)
        set_default_world(w)
        set_default_session(s)
    else:
        app = App.from_config(path, config, llm=llm, ef=ef, logger=logger, **kwargs)
        s = app.world.create_session()
        set_default_session(s)