#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Substrate

Created on Fri Oct 22 14:40:01 2021

Copyright CNRS

@author: david.coulette@ens-lyon.fr

"""

from dementmutant.biomassfield import BiomassField


class Substrate(BiomassField):
    """
    Soil organic substrates concentration fields

    """
    def __init__(self,
                 grid_shape,
                 substrates_names,
                 ):

        """
        The constructor Substrate class.

        Parameters
        ----------
            grid_shape : tuple of int
                (nx,ny) Shape of the spatial grid
            substrate_names  : list of str
                substrate names used for indexing
        """

        super().__init__(grid_shape, substrates_names)

    def get_metadata_dict(self):
        return {
                'grid_shape': self.grid_shape,
                'names': self.names
                }

    def set_parameters_from_metadata(self, pdict, locator, functions_module):
        return
