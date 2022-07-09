# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import, unicode_literals

import setuptools  # required to allow for use of python setup.py develop, may also be important for cython/compiling if it is used

from distutils.core import setup

def extract_version():
    """transition function as try filt which defines version
    in eeghdf/__init__.py"""
    import re
    
    lines = open("eeghdf/__init__.py").readlines()
    flines = [ll for ll in lines if  ll.startswith("__version__")]
    if len(flines)==1:
        rg2 = re.compile(r'''__version__\s*=\s*["'](?P<version>[a-zA-Z0-9_.]+)["']''')
        s = flines[0]
        m = rg2.match(s)
        if m:
            return m.group('version')

    raise Exception("__version__ not defined in eeghdf/__init__.py")


    

setup(
    name="eeghdf",
    version=extract_version(),
    description="""eeg storage in hdf5 + related functions""",
    author="""Chris Lee-Messer""",
    url="https://github.com/eegml/eeghdf",
    # download_url="",
    classifiers=[
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
    ],
    packages=["eeghdf"],
    long_description=open("README.md").read(),
    # package_data={}
    # data_files=[],
    # scripts = [],
    install_requires=["numpy", "h5py", "pandas", "future"],
)
