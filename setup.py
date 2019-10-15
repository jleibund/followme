from setuptools import setup, find_packages
from distutils.core import setup
from Cython.Build import cythonize

setup(name='followme', version='1.0', packages=find_packages(), ext_modules=cythonize("pilots/*.py"))
