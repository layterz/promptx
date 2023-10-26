import os
import click
from rich import pretty, print
import requests
from IPython import embed
from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import load


API_ENDPOINT = os.environ.get('API_ENDPOINT', 'http://localhost:8000')

pretty.install()

@click.group()
def cli():
    pass


@cli.command()
@click.argument('input', nargs=1)
def prompt(input: str):
    _prompt(input)

def _prompt(input: str):
    return requests.post(f'{API_ENDPOINT}/prompt', json={'name': 'n/a', 'instructions': input})


@cli.command(name='query')
@click.argument('texts', nargs=-1)
@click.option('--where', default=None)
@click.option('--field', default=None)
def query(*texts, where=None, field=None):
    _query(*texts, where=where, field=field)

def _query(*texts, where=None, field=None):
    query = { 'texts': texts, 'where': where, 'field': field }
    return requests.post(f'{API_ENDPOINT}/query', data=query)


@cli.command(name='api')
@click.option('--path', default='local')
@click.option('--host', default='0.0.0.0')
@click.option('--port', default='8000')
def api(path=None, host=None, port=None):
    app = load(path)
    import uvicorn
    uvicorn.run(app.api.fastapi_app, host=host, port=port)


@cli.command(name='admin')
@click.option('--path', default='local')
@click.option('--host', default='0.0.0.0')
@click.option('--port', default='8000')
def admin(path=None, host=None, port=None):
    app = load(path)
    from waitress import serve
    serve(app.admin.app.server, host=host, port=port)


@cli.command(name='init')
@click.argument('path', nargs=1, required=False)
def init(path=None):
    # create a hidden directory in the current directory called .px
    if path is None:
        path = os.getcwd()
    elif not os.path.exists(path):
        os.mkdir(path)
    dir = os.path.join(path, '.px')
    if os.path.exists(dir):
        print(f'Error: {dir} already exists')
        return
    os.mkdir(dir)
    print(f'Promptx project created at: {path}')


def main():
    cli()