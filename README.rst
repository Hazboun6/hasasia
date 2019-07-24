===============
hasasia
===============


.. image:: https://img.shields.io/pypi/v/hasasia.svg
        :target: https://pypi.python.org/pypi/hasasia

.. image:: https://img.shields.io/travis/Hazboun6/hasasia.svg
        :target: https://travis-ci.org/Hazboun6/hasasia

.. image:: https://readthedocs.org/projects/hasasia/badge/?version=latest
        :target: https://hasasia.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




A Python package to calculate gravitational-wave sensitivity curves for pulsar timing arrays.

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

Publication
-----------
This work is featured in a publication_, currently released on the ArXiv. If you
would like to reference this work please use the following attribution:

.. _publication: https://arxiv.org/pdf/1907.04341.pdf

.. code-block:: tex

@article{Hazboun:2019vhv,
    author         = "Hazboun, Jeffrey S. and Romano, Joseph D. and Smith,
                      Tristan L.",
    title          = "{Realistic sensitivity curves for pulsar timing arrays}",
    year           = "2019",
    eprint         = "1907.04341",
    archivePrefix  = "arXiv",
    primaryClass   = "gr-qc",
    SLACcitation   = "%%CITATION = ARXIV:1907.04341;%%"
}

Credits
-------
Development Team: Jeffrey S. Hazboun, Joseph D. Romano  and Tristan L. Smith

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
