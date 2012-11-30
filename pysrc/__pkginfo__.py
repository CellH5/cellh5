"""
                           The CellCognition Project
        Copyright (c) 2006 - 2012 Michael Held, Christoph Sommer
                      Gerlich Lab, ETH Zurich, Switzerland
                              www.cellcognition.org

              CellCognition is distributed under the LGPL License.
                        See trunk/LICENSE.txt for details.
                 See trunk/AUTHORS.txt for author contributions.
"""

#-------------------------------------------------------------------------------
# standard library imports:
#
from setuptools import find_packages

#-------------------------------------------------------------------------------
# cecog imports:
#
from cellh5 import (VERSION_NUM,
                   VERSION,
                   )

#-------------------------------------------------------------------------------
# constants:
#
name = 'CellH5Browser'
numversion = VERSION_NUM
version = VERSION
author = 'Christoph Sommer'
author_email = 'sommerc(at)cellcognition.org'
license = 'GPL',
description = ''
long_description = \
"""
"""
url = 'http://www.cellcognition.org'
download_url = ''
package_dir = {}
packages = find_packages()
classifiers = []
platforms = ['Win32']
provides = []
