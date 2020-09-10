#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for tutorial notebooks."""

import pytest
import subprocess as sp

juno_command = 'jupyter nbconvert --to notebook --inplace --execute ../docs/_static/'
def test_sensitivity_tutorial():
    sp.call(juno_command+'sensitivity_tutorial.ipynb',
            shell=True)

def test_skymap_tutorial():
    sp.call(juno_command+'skymap_tutorial.ipynb',
            shell=True)

def test_pulse_profiles_tutorial_3():
    sp.call(juno_command+'real_data_tutorial.ipynb',
            shell=True)
