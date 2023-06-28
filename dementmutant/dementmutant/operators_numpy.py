#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numpy implementation of some operators

Created on Fri Jan 21 13:22:58 2022

Copyright CNRS

@author: david.coulette@ens-lyon.fr

:todo:
    - deprecate ?

"""

import numpy as np


def _apply_delta_sum(mass, delta_mass):
    mass += delta_mass
    return mass
