import setuptools
from setuptools import Extension
from Cython.Build import cythonize


setuptools.setup(
    name='colloptpy',
    version='0.1.0',
    author='Christoph Hoeppke',
    description='Collocation optimisation software written in Python.',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'torch',
        'functorch',
        'pandas'
    ],
)

