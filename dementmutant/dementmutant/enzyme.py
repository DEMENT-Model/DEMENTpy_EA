#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enzymatic entities

Created on Fri Oct 22 11:38:23 2021

Copyright CNRS

@author: david.coulette@ens-lyon.fr

"""

import pandas as pd
import numpy as np

from dementmutant.defaults import _default_dtype
from dementmutant.utility import _draw_array_from_array_linear_constraint
from dementmutant.utility import _draw_array_from_bounds_lhs
from dementmutant.biomassfield import BiomassField


def _Arrhenius_law(vmax, Ea, temperature_celsius, Tref=293.0):
    """
    Arrhenius kinetic factor ( stemming from Boltzmann statistics)

    Parameters
    ----------

        Vmax : numpy.ndarray of floats
            reference value for T = Tref = 293 K = 20°C

        Ea : numpy.ndarray of floats
            activation energy in kJ

        temperature_celsius: float
            Temperature in Celsius degrees

        Tref : float
            Reference tempeature in Kelvin

    Return
    ------
        numpy.ndarray
            Vmax from Arrhenius law

    """
    R = 0.008314  # Universal gas constant in kJ
    Tref = 293.0  # Reference temperature in K (20°C)
    T = temperature_celsius + 273.0
    exp_arg = Ea * (T-Tref) / (R * Tref * T)
    res = vmax * np.exp(exp_arg)
    return res.astype(_default_dtype)


class MMEnzyme:
    """
    Enzyme modeled by Michaelis-Menten kinetic parameters
    Defines parameters for a set of n_enzymes enzymes acting on n_reactions

    Notes :

    We use here simple reactions with 1 reactant so that the identification
    reaction of reactant (substrate or monomer) is often implicitely made.

    """
    _params_dtype = _default_dtype
    _kinetic_parameters_names = ['Ea', 'Vmax', 'Km']
    _association_maps_names = ['map', 'map_t']

    def __init__(
            self,
            enzyme_names,
            reaction_names,
            kin_param_dict={},
            ):
        """
        Parameters
        ----------

            enzyme_names : list of strings
                List of names of enzymatic entities. The number of enzymes is
                inferred from its length
            reactions_names : list of strings
                List of names of reactions ( identified with unique reactants)

            kin_param_dict : dict
                Dictionnary of kinetic parameters
        """
        self.enzyme_names = enzyme_names
        self.n_enzymes = len(enzyme_names)
        self.enz_id = {k: i for i, k in enumerate(self.enzyme_names)}
        self.reaction_names = reaction_names
        self.reac_id = {k: i for i, k in enumerate(self.reaction_names)}
        self.n_reactions = len(self.reaction_names)
        self._kinetic_parameters_shape = (self.n_reactions, self.n_enzymes)
        self.environmental_modulation = {'Vmax':  None, 'Km': None}
        self.environmental_modulation_names = {'Vmax': '', 'Km': ''}
        self.__enzyme_indexes = {k: i for i, k in enumerate(self.enzyme_names)}
        self.__reaction_indexes = {k: i for i, k in enumerate(self.reaction_names)}

        for fname in ['Ea', 'Vmax', 'Km']:
            setattr(self,
                    fname,
                    np.zeros(self._kinetic_parameters_shape, self._params_dtype)
                    )
        self.map = np.ones(self._kinetic_parameters_shape, dtype=bool)
        self.map_t = np.ones(self._kinetic_parameters_shape, self._params_dtype)

        self.set_base_kinetic_params(kin_param_dict)

    def get_enzyme_index(self, enzname):
        """

        Get enzyme index from its name

        Parameters
        ----------
            enzname : string
                Name of enzyme

        Return
        ------
            int
                index of enzyme

        """
        return self.__enzyme_indexes(enzname)

    def get_reaction_index(self, reacname):
        """
        Get index of reaction from its name

        Parameters
        ----------
            reacname : string
                Name of reaction (reactant)

        Return
        ------
            int
                index of reaction

        """
        return self.__reaction_indexes(reacname)

    def set_map_from_bounds(self, bounds, rng):
        """
        Draw reaction/enzymes association with only bounds constraints

        Parameters
        ----------

            bounds : 2-tuple of ints
                Min, Max number of enzymes per reaction

            rng: numpy.random generator
                Random generator to use

        """
        # draw taxon  association
        assert(bounds[0] >= 0)
        assert(bounds[1] <= self.n_enzymes)
        assert(bounds[0] <= bounds[1])
        #
        n_enz = rng.choice(range(bounds[0], bounds[1]+1), size=self.n_reactions)
        #
        self.map[:, :] = False
        for ireac in range(self.map.shape[0]):
            self.map[ireac, :n_enz[ireac]] = True
            rng.shuffle(self.map[ireac, :])
        self.map_t[:, :] = self.map.astype(self.map_t.dtype)

    def set_map_from_func(self, map_func):
        """

        Set the enzyme/substrate (reaction) assocation using a user
        provided function (string, string) --> bool. Given a pair
        (substrate name, enzyme name), this function must return
        True if the enzyme acts as an enzymatic modulator
        in the substrate degradation

        Parameters
        ----------
            map_func : function (string, string) --> bool

        """
        for isub, sub in enumerate(self.reaction_names):
            for ienz, enz in enumerate(self.enzyme_names):
                self.map[isub, ienz] = map_func(sub, enz)
        self.map_t = self.map.astype(_default_dtype)

    def get_Ea_input_from_file(self, filename):
        """
            Load activation energies from csv file as a panda Dataframe

        Parameters
        ----------
            filename : string
                Path of the file to read

        Return
        ------
            pandas.DataFrame
                DataFrame of Activation energies bounds

        """
        res = pd.read_csv(
                          filename,
                          index_col=0,
                          dtype={k: _default_dtype for k in ['Ea_min', 'Ea_max']}
                          )
        return res

    def draw_Ea_from_bounds_dict_uniform(self, Ea_bounds_dict, rng):
        """
            Draw activation energies from bounds provided as a dictionnary

        Parameters
        ----------
            Ea_bounds_dict : dict
                Dictionnary of activation energy bounds

            rng : numpy.random generator
                Random generator to use

        """
        if all(c in Ea_bounds_dict.keys() for c in ['Ea_min', 'Ea_max']):
            bounds = Ea_bounds_dict
            draws = rng.uniform(bounds['Ea_min'], bounds['Ea_max'], size=self.Ea.shape)
            self.Ea[()] = draws[()]
        elif all(c in Ea_bounds_dict.keys() for c in self.reaction_names):
            for reac, bounds in Ea_bounds_dict.items():
                draws = rng.uniform(bounds['Ea_min'],
                                    bounds['Ea_max'],
                                    size=self.n_enzymes)
                self.Ea[self.read_id[reac], :] = draws
        else:
            print('Improper bounds provided')

    def draw_Ea_from_bounds_df_uniform(self, Ea_bounds_df, rng):
        """
            Draw activation energies from bounds provided as a pandas.DataFrame

        Parameters
        ----------
            Ea_bounds_dict : dict
                DataFrame of activation energy bounds
                rows : reactions, cols : Ea_min, Ea_max

            rng : numpy.random generator
                Random generator to use

        """

        assert(all([c in Ea_bounds_df.columns for c in['Ea_min', 'Ea_max']]))

        full_match = (
                      len(Ea_bounds_df.index) == len(self.reaction_names)
                      and
                      all(c in self.reaction_names for c in Ea_bounds_df.index)
                     )
        if full_match:
            for reac in Ea_bounds_df.index:
                bounds = Ea_bounds_df.loc[reac]
                draws = rng.uniform(bounds['Ea_min'],
                                    bounds['Ea_max'], size=self.n_enzymes)
                self.Ea[self.reac_id[reac], :] = draws
        elif (len(Ea_bounds_df.index) == 1):
            bounds = Ea_bounds_df.loc[0]
            draws = rng.uniform(bounds['Ea_min'], bounds['Ea_max'], size=self.Ea.shape)
            self.Ea[()] = draws[()]
        else:
            print('Improper bounds provided')

    def get_map_bool_df(self):
        """
        Return a copy the reaction/enzyme boolean association matrix
        as a pandas.DataFrame

        Return
        ------
            pandas.DataFrame
                Reaction/Enzyme association matrix

        """
        df = pd.DataFrame(
                data=self.map,
                index=self.reaction_names,
                columns=self.enzyme_names,
                copy=True
                )
        return df

    def get_base_kinetic_parameter_df(self, paramname):
        """
        Return base kinetic parameter as a pandas.DataFrame

        Parameters
        ----------
            paramname : string
                Name of the required kinetic parameter ('Vmax', 'Km')
        Return
        ------
            pandas.DataFrame
                Dataframe of values of the kinetic parameter
                rows : reactions , cols : enzymes
        """
        if (not(paramname in self._kinetic_parameters_names)):
            print('Unknown kinetic parameter {}'.format(paramname))
            return None
        df = pd.DataFrame(
                data=getattr(self, paramname),
                index=self.reaction_names,
                columns=self.enzyme_names,
                copy=True,
                )
        return df

    def draw_VmaxKm_linear_tradeoff(self,
                                    Vmax_bounds,
                                    Km_rel_error,
                                    Vmax_Km_slope,
                                    Vmax_Km_intercept,
                                    Km_bounds,
                                    rng,
                                    specificity_tradeoff=0,
                                    ):
        """
        Draw kinetic parameters using
            - lhs law for Vmax
            - linear Vmax/Km trade-off constraint for Km

        Parameters
        ----------

            Vmax_bounds : tuple of floats
                bounds for Vmax

            Km_rel_error : float
                relative spread of Km wrt linear Vmax law

            Vmax_Km_slope : float
                slope of reference linear law

            Vmax_Km_intercept: float
                intercept of reference linear law

            Km_bounds : tuple of floats
                A posteriori bounds to apply to Km

            rng : numpy.random generator
                Random generator to use

            specificity_tradeoff : float
                specificity tradeoff factor

        """
        self.Vmax = _draw_array_from_bounds_lhs(self.Vmax, Vmax_bounds, rng)
        if (specificity_tradeoff > 0):
            nnz = specificity_tradeoff * np.sum(self.map_t, axis=0)
            nnz[nnz == 0] = 1.0
            self.Vmax = self.Vmax / nnz[np.newaxis, :]

        self.Vmax = self.Vmax * self.map_t

        self.Km = _draw_array_from_array_linear_constraint(
                                                     self.Km,
                                                     self.Vmax,
                                                     Km_rel_error,
                                                     Vmax_Km_slope,
                                                     Vmax_Km_intercept,
                                                     Km_bounds,
                                                     rng,
                                                     )

    def set_environmental_modulation(self, kinparam, func):
        assert(kinparam in self.environmental_modulation.keys())
        self.environmental_modulation[kinparam] = func
        self.environmental_modulation_names[kinparam] = func.__name__

    def set_base_kinetic_params(self, kin_param_dict):
        for kp_name, d in kin_param_dict.items():
            o = getattr(self, kp_name)
            o[()] = d[()]

    def get_kinetic_parameters(self, env):

        res = {}
        for kpname in ['Vmax', 'Km']:
            tmp_f = _Arrhenius_law(getattr(self, kpname), self.Ea, env['temp'])
            if (self.environmental_modulation[kpname] is not None):
                fact = self.environmental_modulation[kpname](env)
                fact = _default_dtype(fact)
                tmp_f = tmp_f * fact
                tmp_f = tmp_f
            res[kpname] = tmp_f

        return res['Vmax'], res['Km']

    def get_metadata_dict(self):
        direct_attrs = ['enzyme_names',
                        'reaction_names',
                        'environmental_modulation_names'
                        ]
        res = {}
        for a in direct_attrs:
            res[a] = getattr(self, a)
        np_params = ['map', ] + self._kinetic_parameters_names
        for a in np_params:
            res[a] = getattr(self, a).tolist()

        return res

    def set_parameters_from_metadata(self, pdict, locator, functions_module):

        np_params = ['map', ] + self._kinetic_parameters_names

        for a in np_params:
            o = getattr(self, a)
            o[()] = np.array(pdict[a])

        self.environmental_modulation_names = pdict['environmental_modulation_names']
        for k,fname in self.environmental_modulation_names.items():
            if (len(fname) > 0):
                f = getattr(functions_module, fname)
            else:
                f = None
            self.environmental_modulation[k] = f

class DegradEnzyme(MMEnzyme, BiomassField):
    """
    Class for sustrate degradation enzyme
    """
    def __init__(self,
                 grid_shape,
                 enzyme_names,
                 substrate_names,
                 ):
        """

        Parameters
        ----------

            grid_shape : 2-tuple of ints
                Shape of space grid

            enzyme_names : list of string
                List of enzyme names

            substrate_names : list of string
                List of substrate_names (reactants). Degradation reactions
                are identified with substrates, as there is one reaction per
                substrate

        """
        BiomassField.__init__(self, grid_shape, enzyme_names)
        MMEnzyme.__init__(self, enzyme_names, substrate_names)

    def get_metadata_dict(self):
        res = MMEnzyme.get_metadata_dict(self)
        res['grid_shape'] = self.grid_shape
        return res


class UptakeTransporter(MMEnzyme):
    """
        Class for Uptake Transporters "enzymes"
    """
    def __init__(self,
                 transporter_names,
                 monomer_names,
                 ):
        """
        Parameters
        ----------

            transporter_names : list of string
                List of names of uptake transporter

            monomer_names : list of string
                List of names of monomers

        """
        MMEnzyme.__init__(self, transporter_names, monomer_names)
