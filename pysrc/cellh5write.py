import warnings
warnings.simplefilter("always")
warnings.warn("cellh5write module moved into the cellh5 module and will be removed in version 1.4.0", DeprecationWarning)

from cellh5.cellh5write import *