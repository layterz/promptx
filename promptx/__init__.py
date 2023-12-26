import os

from .collection import Collection, MemoryVectorDB
from .world import Session
from .application import App

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


def load(path=None, **env):
    try:
        from .adapters.chromadb import ChromaVectorDB
        db = ChromaVectorDB()
    except ImportError:
        db = MemoryVectorDB()

    try:
        from .models.openai import ChatGPT
        api_key = os.environ.get('OPENAI_API_KEY')
        org_id = os.environ.get('OPENAI_ORG_ID')
        llm = ChatGPT(id='default', api_key=api_key, org_id=org_id)
    except ImportError:
        from .models import MockLLM
        llm = MockLLM()

    app = None
    if path is None:
        project_path = find_project_root()
        if project_path is not None:
            path = project_path

    env = {**os.environ, **(env or {})}
    app = App.load(path, db, llm, env=env)
    s = app.world.create_session()
    set_default_session(s)
    return app


import os

if os.environ.get('PXX_AUTOLOAD', True):
    load()