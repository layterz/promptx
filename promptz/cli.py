import os
import shutil
import click
import requests
from IPython import embed
from jinja2 import Environment, FileSystemLoader, select_autoescape


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