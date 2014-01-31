"""
version.py
"""

__author__ = 'rudolf.hoefler@gmail.com'
__copyright__ = ('The CellH5 Project'
                 'Copyright (c) 2012 - 2013 Christoph Sommer, Rudolf Hoefler, '
                 'Michael Held, Bernd Fischer'
                 'Gerlich Lab, IMBA Vienna, Huber Lab, EMBL Heidelberg')

__licence__ = 'LGPL'
__url__ = 'www.cellh5.org'


major = 1
minor= 1
release = 0

version_num = (major, minor, release)
version = '.'.join([str(n) for n in version_num])
