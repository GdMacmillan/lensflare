# -*- encoding: utf-8 -*-
import os
import sys

from lensflare.util import dependencies
from lensflare.__version__ import __version__

__MANDATORY_PACKAGES__ = '''
numpy>=1.9
tensorflow>=1.1
matplotlib>=2.1
'''

dependencies.verify_packages(__MANDATORY_PACKAGES__)

if os.name != 'posix':
    raise ValueError(
        'Detected unsupported operating system: %s. Please check '
        'the compability information of lensflares' %
        sys.platform
    )

if sys.version_info < (3, 5):
    raise ValueError(
        'Unsupported python version %s found. Auto-sklearn requires Python '
        '3.5 or higher.' % sys.version_info
    )
