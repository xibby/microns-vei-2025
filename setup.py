#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='staticnet',
    version='0.0.1',
    description='PyTorch implementation of static neural networks for system identification',
    author='Fabian Sinz, Edgar. Y. Walker',
    author_email='sinz@bcm.edu',
    packages=find_packages(exclude=[]),
    install_requires=['numpy', 'scipy>=1.2', 'tqdm', 'gitpython', 'scikit-image',
                      'datajoint', 'h5py', ],
)
