import os
import sys
import shutil
import click
import requests
import pprint
from IPython import embed
from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import load
from promptz.application import App


def create_project_structure(template_path, project_path, variables):
    # Initialize a Jinja2 environment
    env = Environment(loader=FileSystemLoader(template_path), autoescape=select_autoescape(['html', 'xml']))

    for root, dirs, files in os.walk(template_path):
        # Create the directories in the new location
        for dir_name in dirs:
            dest_dir = os.path.join(project_path, os.path.relpath(root, template_path), dir_name)
            os.makedirs(dest_dir, exist_ok=True)

        # Copy the files to the new location, rendering them if necessary
        for file_name in files:
            # If the file is a Jinja2 template
            if file_name.endswith('.j2'):
                # Remove the .j2 extension to get the output file name
                dest_file = os.path.join(project_path, os.path.relpath(root, template_path), file_name[:-3])

                # Load the template
                template = env.get_template(os.path.relpath(os.path.join(root, file_name), template_path))

                # Render the template with the provided variables and write it to the file
                with open(dest_file, 'w') as f:
                    f.write(template.render(variables))
            else:
                # If the file is not a Jinja2 template, just copy it
                src_file = os.path.join(root, file_name)
                dest_file = os.path.join(project_path, os.path.relpath(root, template_path), file_name)
                shutil.copy(src_file, dest_file)


API_ENDPOINT = os.environ.get('API_ENDPOINT', 'http://localhost:8000')

@click.group()
def cli():
    pass

@cli.command(name="create")
@click.argument('project_name')
@click.option('--template', default='default')
def create_project(project_name, template='default'):
    current_dir = os.path.dirname(__file__)
    template_dir = os.path.join(current_dir, f'templates/{template}')
    create_project_structure(template_dir, project_name, {'project_name': project_name})


@cli.command()
@click.argument('input', nargs=1)
def prompt(input: str):
    _prompt(input)

def _prompt(input: str):
    return requests.post(f'{API_ENDPOINT}/prompt', json={'name': 'n/a', 'instructions': input})


@cli.group()
def templates():
    pass

@templates.command(name="list")
def list_templates():
    r = requests.get(f'{API_ENDPOINT}/templates')
    pprint.pprint(r.json()['response'])

@templates.command(name="run")
@click.argument('name')
@click.argument('input')
def run_template(name, input):
    _run_template(name, input)

def _run_template(name, input):
    return requests.post(f'{API_ENDPOINT}/templates/{name}/run', data=input)

@templates.command(name="create")
@click.argument('name')
@click.argument('instructions')
def create_template(name, instructions):
    return requests.post(f'{API_ENDPOINT}/templates', 
                         json={'name': name, 'instructions': instructions})

@cli.command(name='query')
@click.argument('texts', nargs=-1)
@click.option('--where', default=None)
@click.option('--field', default=None)
def query(*texts, where=None, field=None):
    _query(*texts, where=where, field=field)

def _query(*texts, where=None, field=None):
    query = { 'texts': texts, 'where': where, 'field': field }
    return requests.post(f'{API_ENDPOINT}/query', data=query)


@cli.group()
def systems():
    pass

@systems.command(name="list")
def list_systems():
    return requests.get(f'{API_ENDPOINT}/systems')

@systems.command(name="run")
@click.argument('name')
def run_system(name):
    return requests.post(f'{API_ENDPOINT}/systems/run')


@cli.command(name='serve')
@click.option('--path', default='local')
def serve(path=None):
    app = load(path)
    app.serve()


@cli.command()
@click.argument('input', nargs=-1)
def root(input: str):
    _prompt(input)


def main():
    cli()