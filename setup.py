#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from setuptools import setup


VERSION_TEMPLATE = """
# Note that we need to fall back to the hard-coded version if either
# setuptools_scm can't be imported or setuptools_scm can't determine the
# version, so we catch the generic 'Exception'.
try:
    from setuptools_scm import get_version
    __version__ = get_version(root='..', relative_to=__file__)
except Exception:
    __version__ = '{version}'
""".lstrip()

setup(
    use_scm_version={'write_to': os.path.join('pyonset', 'version.py'),
                     'write_to_template': VERSION_TEMPLATE},

)

# setup(
#     name="pyonset",
#     version="0.0.1",    
#     description="A python SEP event analysis software",
#     url="https://github.com/Christian-Palmroos/PyOnset",
#     author="Christian Palmroos",
#     author_email="chospa@utu.fi",
#     license="BSD 2-clause",
#     packages=["pyonset"],
#     install_requires=["numpy",
#                       "pandas",
#                       "scipy"                     
#                       ],

#     classifiers=[
#         "Development Status :: 1 - Planning",
#         "Intended Audience :: Science/Research",
#         "License :: OSI Approved :: BSD License",  
#         "Operating System :: POSIX :: Linux",        
#         "Programming Language :: Python :: 3",
#         "Programming Language :: Python :: 3.10",
#     ],
# )
