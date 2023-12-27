from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='pxx',
    version='0.0.8',
    description='An AI framework',
    packages=['promptx'],
    install_requires=requirements,
    extras_require={
        'chromadb': ['chromadb==0.4.14'],
        'openai': ['openai'],
    },
)