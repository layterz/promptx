from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='pxx',
    version='0.0.9',
    description='An AI framework',
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        'chromadb': ['chromadb==0.4.14'],
        'openai': ['openai'],
    },
)