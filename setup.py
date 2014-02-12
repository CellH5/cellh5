"""
setup.py

Setup script for cellh5.py

This script installs only the library

>>>import cellh5
"""

__author__ = 'rudolf.hoefler@gmail.com'


from distutils.core import setup
import version


setup(name='cellh5',
      version = version.version,
      description = 'module for easy acces of cellh5 files',
      author = 'Rudolf Hoefler',
      author_email = 'rudolf.hoefler@gmail.com',
      license = 'LGPL',
      url = 'http://cellh5.org',
      package_dir = {'': 'pysrc'},
      py_modules = ['cellh5'])
