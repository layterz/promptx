import os
import click

@click.group()
def cli():
    pass

@cli.command(name="create")
@click.argument('project_name')
def create_project(project_name):
    create_project_structure(project_name)

def create_project_structure(project_name):
    directories = [
        f"{project_name}/{project_name}",
        f"{project_name}/prompts",
        f"{project_name}/systems",
        f"{project_name}/models",
        f"{project_name}/admin",
        f"{project_name}/api",
        f"{project_name}/notebooks",
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # Create __init__.py file in each directory
        with open(os.path.join(directory, "__init__.py"), "w") as init_file:
            init_file.write("")

@cli.group()
def prompts():
    pass

@prompts.command()
@click.argument('input', nargs=-1)
def prompt(input: str):
    _prompt(input)

def _prompt(input: str):
    print(f'Prompting with {input}')

@prompts.command(name="create")
@click.argument('name')
def create_prompt(name):
    _create_prompt(name)

def _create_prompt(name):
    with open(f"prompts/{name}.py", "w") as prompt_file:
        prompt_file.write(f"def {name}():\n    pass\n")

@prompts.command(name="list")
def list_prompts():
    _list_prompts()

def _list_prompts():
    for prompt in os.listdir("prompts"):
        print(prompt)

@prompts.command(name="run")
@click.argument('name')
def run_prompt(name):
    _run_prompt(name)

def _run_prompt(name):
    print(f'Running prompt {name}')

@cli.command(name='query')
@click.argument('texts', nargs=-1)
@click.option('--where', default=None)
@click.option('--field', default=None)
def query(*texts, where=None, field=None):
    _query(*texts, where=where, field=field)

def _query(*texts, where=None, field=None):
    print(f'Querying {texts}')


@cli.group()
def models():
    pass

@models.command(name="create")
@click.argument('name')
def create_model(name):
    _create_model(name)

def _create_model(name):
    with open(f"models/{name}.py", "w") as model_file:
        model_file.write(f"def {name}():\n    pass\n")

@models.command(name="list")
def list_models():
    _list_models()

def _list_models():
    for model in os.listdir("models"):
        print(model)


@cli.group()
def systems():
    pass

@systems.command(name="create")
@click.argument('name')
def create_system(name):
    _create_system(name)

def _create_system(name):
    with open(f"systems/{name}.py", "w") as system_file:
        system_file.write(f"def {name}():\n    pass\n")

@systems.command(name="list")
def list_systems():
    _list_systems()

def _list_systems():
    for system in os.listdir("systems"):
        print(system)

@systems.command(name="run")
@click.argument('name')
def run_system(name):
    _run_system(name)

def _run_system(name):
    print(f'Running system {name}')


@cli.command(name='repl')
def repl():
    _repl()

def _repl():
    print(f'Opening REPL')


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