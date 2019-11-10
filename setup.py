
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='GraphletMiner',
    version='0.1.0',
    description='Graphlet miner: a text pattern analysis library for Python',
    long_description=readme,
    author='Ahmed Nabhan',
    author_email='ahmed.nabhan@gmail.com',
    url='https://github.com/arnabhan/graphletminer',
    license=license,
    install_requires=['nltk','networkx','tqdm'],
    packages=find_packages(exclude=('tests', 'docs'))
)

