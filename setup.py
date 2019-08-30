#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy',
                'scipy',
                'astropy',]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Jeffrey S. Hazboun",
    author_email='jeffrey.hazboun@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="A Python package to calculate gravitational-wave sensitivity curves for pulsar timing arrays.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='hasasia',
    name='hasasia',
    packages=find_packages(include=['hasasia']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/Hazboun6/hasasia',
    version='0.1.7',
    zip_safe=False,
)
