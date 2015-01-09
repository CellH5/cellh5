"""
setup.py

Setup script for cellh5.py

This script installs only the library

>>>import cellh5
"""

__author__ = 'rudolf.hoefler@gmail.com'

import sys
sys.path.append('pysrc')

from distutils.core import setup
import cellh5

setup(name='cellh5',
      version = cellh5,
      description = 'module for easy acces of cellh5 files',
      author = 'Christoph Sommer, Rudolf Hoefler',
      author_email = 'christoph.sommer@imba.oeaw.ac.at, rudolf.hoefler@gmail.com',
      license = 'LGPL',
      url = 'http://cellh5.org',
      package_dir = {'': 'pysrc'},
      py_modules = ['cellh5'])
