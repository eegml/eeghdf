# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import, unicode_literals

import setuptools  # required to allow for use of python setup.py develop, may also be important for cython/compiling if it is used

from distutils.core import setup


setup(
    name="eeghdf",
    version="0.1",
    description="""eeg storage in hdf5 + related functions""",
    author="""Chris Lee-Messer""",
    url="https://github.com/cleemesser/eeghdf",
    # download_url="",
    classifiers=["Topic :: Science :: EEG"],
    packages=["eeghdf"],
    long_description=open('README.md').read(),
    # package_data={}
    # data_files=[],
    # scripts = [],
)
