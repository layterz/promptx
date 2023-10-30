from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='pxx',
    version='0.0.2',
    description='An AI framework',
    packages=['promptx'],
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "px = promptx.cli:main",
        ],
    },
)