import os
import click
import requests
from IPython import embed


API_ENDPOINT = os.environ.get('API_ENDPOINT', 'http://localhost:8000')

@click.group()
def cli():
    pass

@cli.command(name="create")
@click.argument('project_name')
def create_project(project_name):
    create_project_structure(project_name)

def create_project_structure(project_name):
    directories = [
        f"{project_name}/prompts",
        f"{project_name}/systems",
        f"{project_name}/admin",
        f"{project_name}/notebooks",
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    with open(os.path.join(f'{project_name}/systems', "__init__.py"), "w") as init_file:
        init_file.write('')
    
    with open(os.path.join(project_name, "app.py"), "w") as app_file:
        app_file.write(f'''from promptz import App

def create_app():
    app = App(name="{project_name}")
    return app''')

@cli.group()
def prompts():
    pass

@prompts.command()
@click.argument('input', nargs=-1)
def prompt(input: str):
    _prompt(input)

def _prompt(input: str):
    return requests.post(f'http://{API_ENDPOINT}/prompt', data=input)

@prompts.command(name="list")
def list_prompts():
    _list_prompts()

def _list_prompts():
    return requests.get(f'http://{API_ENDPOINT}/prompts')

@prompts.command(name="run")
@click.argument('name')
@click.argument('input')
def run_prompt(name, input):
    _run_prompt(name, input)

def _run_prompt(name, input):
    return requests.post(f'http://{API_ENDPOINT}/prompts/{name}/run', data=input)

@cli.command(name='query')
@click.argument('texts', nargs=-1)
@click.option('--where', default=None)
@click.option('--field', default=None)
def query(*texts, where=None, field=None):
    _query(*texts, where=where, field=field)

def _query(*texts, where=None, field=None):
    query = { 'texts': texts, 'where': where, 'field': field }
    return requests.post(f'http://{API_ENDPOINT}/query', data=query)


@cli.group()
def systems():
    pass

@systems.command(name="list")
def list_systems():
    _list_systems()

def _list_systems():
    return requests.get(f'http://{API_ENDPOINT}/systems')

@systems.command(name="run")
@click.argument('name')
def run_system(name):
    _run_system(name)

def _run_system(name):
    # call the /systems/run endpoint
    return requests.post(f'http://{API_ENDPOINT}/systems/run')


@cli.command(name='repl')
def repl():
    _repl()

def _repl():
    from promptz import World
    world = World('test')
    session = world.create_session()
    query = session.query
    store = session.store
    prompt = session.prompt
    history = session.history
    evaluate = session.evaluate
    collection = session.collection
    chain = session.chain
    run = session.run
    embed(
        header='promptz',
    )


@cli.command(name='serve')
def serve():
    _serve()

def _serve():
    print(f'Serving')


@cli.command()
@click.argument('input', nargs=-1)
def root(input: str):
    _prompt(input)


def main():
    cli()