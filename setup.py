from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='promptz',
    version='0.0.9',
    description='A Python package for interactive prompts',
    packages=['promptz'],
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "pz = promptz.cli:main",
        ],
    },
)