#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Package wide parameters

Created on Thu Dec 16 14:32:49 2021

Copyright CNRS

@author: david.coulette@ens-lyon.fr

"""

import numpy as np

_default_dtype = np.float64
_biomass_atoms_names = ['C', 'N', 'P']

_substrates_default = [
                       'DeadMic',
                       'DeadEnz',
                       'Cellulose',
                       'Hemicellulose',
                       'Starch',
                       'Chitin',
                       'Lignin',
                       'Protein1',
                       'Protein2',
                       'Protein3',
                       'OrgP1',
                       'OrgP2',
                       ]


_organic_monomers_default = _substrates_default

_inorganic_monomers_default = ['NH4', 'PO4']

_subtrates_monomer_association_default = {s: s for s in _substrates_default}

_inorganic_monomers_stoechio_default = {
                                        'NH4': {'C': 0.0, 'N': 1.0, 'P': 0.0},
                                        'PO4': {'C': 0.0, 'N': 0.0, 'P': 1.0},
                                        }

