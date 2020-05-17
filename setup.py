#!/usr/bin/env python3
from setuptools import setup

setup(
    name='brainreader',
    version='0.0.2',
    description='Reconstruct images from recorded two-photon activity',
    author='Erick Cobos',
    author_email='ecobos@bcm.edu',
    license='MIT',
    url='https://github.com/ecobost/brainreader',
    keywords= 'brainreader decoding two-photon',
    packages=['brainreader'],
    install_requires=['datajoint>=0.12', 'torch>=1.0', 'numpy', 'scikit-learn'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English'
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)
