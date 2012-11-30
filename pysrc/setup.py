"""
                           The CellCognition Project
        Copyright (c) 2006 - 2012 Michael Held, Christoph Sommer
                      Gerlich Lab, ETH Zurich, Switzerland
                              www.cellcognition.org

              CellCognition is distributed under the LGPL License.
                        See trunk/LICENSE.txt for details.
                 See trunk/AUTHORS.txt for author contributions.
"""

__author__ = 'Michael Held'
__date__ = '$Date$'
__revision__ = '$Rev$'
__source__ = '$URL$'

#-------------------------------------------------------------------------------
# standard library imports:
#
from setuptools import setup
import shutil
import os
import sys
import matplotlib

#-------------------------------------------------------------------------------
# cecog imports:
#


#-------------------------------------------------------------------------------
# constants:
#
MAIN_SCRIPT = '../apps/cellh5browser/cellh5browser.py'

APP = [MAIN_SCRIPT]
INCLUDES = ['sip', 'tabdelim',]
EXCLUDES = ['PyQt4.QtDesigner', 'PyQt4.QtNetwork',
            'PyQt4.QtOpenGL', 'PyQt4.QtScript',
            'PyQt4.QtSql', 'PyQt4.QtTest',
            'PyQt4.QtWebKit', 'PyQt4.QtXml',
            'PyQt4.phonon',
            'rpy',
            '_gtkagg', '_tkagg', '_agg2', '_cairo', '_cocoaagg',
            '_fltkagg', '_gtk', '_gtkcairo',
            'Tkconstants', 'Tkinter', 'tcl', 'zmq'
            ]
PACKAGES = ['h5py', 'vigra', 'matplotlib']


#-------------------------------------------------------------------------------
# functions:
#
def tempsyspath(path):
    def decorate(f):
        def handler():
            sys.path.insert(0, path)
            value = f()
            del sys.path[0]
            return value
        return handler
    return decorate

def read_pkginfo_file(setup_file):
    path = os.path.dirname(setup_file)
    @tempsyspath(path)
    def _import_pkginfo_file():
        if '__pgkinfo__' in sys.modules:
            del sys.modules['__pkginfo__']
        return __import__('__pkginfo__')
    return _import_pkginfo_file()

# -------------------------------------------------------------------------------
# main:

pkginfo = read_pkginfo_file(__file__)

if sys.platform == 'win32':
    import py2exe # pylint: disable-msg=F0401,W0611
    FILENAME_ZIP = 'data.zip'
    OPTIONS = {'windows': [{'script': MAIN_SCRIPT,
                            'icon_resources': \
                               [(1, 'cellh5_icon.ico')],
                           }],
               # FIXME: the one-file version is currently not working!
               'zipfile' : FILENAME_ZIP,
               }
    SYSTEM = 'py2exe'
    DATA_FILES = matplotlib.get_py2exe_datafiles()
    EXTRA_OPTIONS = {'includes': INCLUDES,
                     'excludes': EXCLUDES,
                     'packages': PACKAGES,
                     'optimize': 2,
                     'compressed': False,
                     'skip_archive': True,
                     'bundle_files': 3,

                     #'ascii': True,
                     #'xref': True,
                    }

setup(
    data_files=DATA_FILES,
    options={SYSTEM: EXTRA_OPTIONS},
    includes=[],
    setup_requires=[SYSTEM],
    name=pkginfo.name,
    version=pkginfo.version,
    author=pkginfo.author,
    author_email=pkginfo.author_email,
    license=pkginfo.license,
    description=pkginfo.description,
    long_description=pkginfo.long_description,
    url=pkginfo.url,
    download_url=pkginfo.download_url,
    classifiers=pkginfo.classifiers,
    package_dir=pkginfo.package_dir,
    packages=pkginfo.packages,
    platforms=pkginfo.platforms,
    provides=pkginfo.provides,
    **OPTIONS
)