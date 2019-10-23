===========
``hasasia``
===========


.. image:: https://img.shields.io/pypi/v/hasasia.svg
        :target: https://pypi.python.org/pypi/hasasia

.. image:: https://img.shields.io/travis/Hazboun6/hasasia.svg
        :target: https://travis-ci.org/Hazboun6/hasasia

.. image:: https://readthedocs.org/projects/hasasia/badge/?version=latest
        :target: https://hasasia.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://zenodo.org/badge/178294587.svg
        :target: https://zenodo.org/account/settings/github/repository/Hazboun6/hasasia
        :alt: Zenodo Badge

.. image:: https://joss.theoj.org/papers/d99d7655bd5704ab951157a14df227af/status.svg
        :target: https://joss.theoj.org/papers/d99d7655bd5704ab951157a14df227af
        :alt: JOSS Status

A Python package to calculate gravitational-wave sensitivity curves for pulsar timing arrays.

.. image:: https://raw.githubusercontent.com/Hazboun6/hasasia/master/hasasia_calligraphy.jpg
        :align: center

حساسية (hasasia) is Arabic for sensitivity_ . Image Credit: Reem Tasyakan

.. _sensitivity: https://translate.google.com/#view=home&op=translate&sl=auto&tl=ar&text=sensitivity

* Free software: MIT license
* Documentation: https://hasasia.readthedocs.io.


Features
--------
Calculates the following structures needed for signal analysis with pulsars:

* Pulsar transmission functions
* Inverse-noise-weighted transmission functions
* Individual pulsar sensitivity curves.
* Pulsar timing array sensitivity curves as characteristic strain, strain sensitivity or energy density.
* Power-law integrated sensitivity curves.
* Sensitivity sky maps for pulsar timing arrays

Getting Started
---------------

`hasasia` is on the Python Package Inventory, so the easiest way to get started
is by using `pip` to install::

  pip install hasasia

The pulsar and spectrum objects are used to build sensitivity curves for full
PTAs. The Spectrum object has all of the information needed for the pulsar.

.. code-block:: python

  import hasasia.senstivity as hsen

  toas = np.arange(54378,59765,22) #Choose a range of times-of-arrival
  toaerrs = 1e-7*np.ones_like(toas) #Set all errors to 100 ns
  psr = hsen.Pulsar(toas=toas,toaerrs=toaerrs)
  spec = hsen.Spectrum(psr)


Publication
-----------
This work is featured in a publication_, currently released on the arXiv. If you
would like to reference the formalism used in this work please use the following attribution:

.. _publication: https://arxiv.org/pdf/1907.04341.pdf

.. code-block:: tex

  @article{Hazboun:2019vhv,
           author         = "Hazboun, Jeffrey S. and Romano, Joseph D. and Smith, Tristan L.",
           title          = "{Realistic sensitivity curves for pulsar timing arrays}",
           year           = "2019",
           eprint         = "1907.04341",
           archivePrefix  = "arXiv",
           primaryClass   = "gr-qc",
           SLACcitation   = "%%CITATION = ARXIV:1907.04341;%%"
           }

Otherwise if you would like to reference the Python package use the following
citation:

.. code-block:: tex
  @article{Hazboun2019Hasasia,
         	 journal      = {Journal of Open Source Software},
         	 doi          = {10.21105/joss.01775},
         	 issn         = {2475-9066},
         	 number       = {42},
         	 publisher    = {The Open Journal},
           title        = {Hasasia: A Python package for Pulsar Timing Array Sensitivity Curves},
           url          = {http://dx.doi.org/10.21105/joss.01775},
         	 volume       = {4},
         	 author       = {Hazboun, Jeffrey and Romano, Joseph and Smith, Tristan},
         	 pages        = {1775},
         	 date         = {2019-10-23},
         	 year         = {2019},
         	 month        = {10},
         	 day          = {23},
           }



Credits
-------
Development Team: Jeffrey S. Hazboun, Joseph D. Romano  and Tristan L. Smith

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
