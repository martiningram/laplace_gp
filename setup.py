from os import getenv
from setuptools import setup
from setuptools import find_packages


setup(
    name='Sparse Laplace GP',
    version=getenv("VERSION", "LOCAL"),
    description='Experiments to fit Gaussian Processes with '
    'Sparse Linear Algebra',
    packages=['sparse_gp']
)
