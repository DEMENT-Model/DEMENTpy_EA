#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monomer related data structures

Created on Fri Oct 22 14:40:01 2021

Copyright CNRS

@author: david.coulette@ens-lyon.fr

------------------------------------------------------------------------------

"""

from dementmutant.biomassfield import BiomassField


class Monomer(BiomassField):
    """
    Monomers , both organic (stemming from substrate degradation) and inorganic.

    """

    def __init__(self,
                 grid_shape,
                 monomer_names,
                 ):
        """
        Parameters
        ----------
            grid_shape : tuple of int
                Shape of the spatial grid
            monomer_names: list of str
                List of monomer names

        """

        super().__init__(grid_shape, monomer_names)

    def get_metadata_dict(self):
        return {
                'grid_shape': self.grid_shape,
                'names': self.names
                }

    def set_parameters_from_metadata(self, pdict, locator, functions_module):
        return
