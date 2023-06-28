#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Main module for ecosystem modeling

Created on Fri Oct 22 11:38:23 2021

Copyright CNRS

author : David Coulette david.coulette@ens-lyon.fr

------------------------------------------------------------------------------

"""

import importlib
import os
import sys
import numpy as np
import pandas as pd
import inspect
import h5py
import json
import shutil

from dementmutant.utility import dict_from_json, StageTimer
from dementmutant.substrate import Substrate
from dementmutant.monomer import Monomer
from dementmutant.enzyme import DegradEnzyme
from dementmutant.enzyme import UptakeTransporter
from dementmutant.microbe import Microbe, StoechiometricBalanceRecycler, MicrobialMortality
from dementmutant.biomassfield import BiomassField

from dementmutant.operators import _apply_delta_sum
from dementmutant.operators import _sub_degrad_decay_rate
from dementmutant.operators import _mon_uptake_decay_rate
from dementmutant.operators import _metabolic_process_ranked
from dementmutant.operators import _apply_mortality_stoech
from dementmutant.operators import _apply_mortality
from dementmutant.operators import _mass_ratio_removal
from dementmutant.operators import _mass_nonlocal_sharing_removal
from dementmutant.operators import _set_shifts_filtered
from dementmutant.operators import _move_delta_mass, _move_delta_mass_withmut
from dementmutant.operators import _apply_decay
from dementmutant.operators import _apply_quantized_birth_mutation
from dementmutant.operators import _apply_quantized_birth_mutation2

from dementmutant.operators import _apply_quantized_mortality_stoech
from dementmutant.operators import _apply_quantized_mortality
from dementmutant.operators import _move_quanta_mass_to_delta_filtered
from dementmutant.operators import  _move_quanta_mass_to_delta_filtered_taxflux
from dementmutant.operators import _apply_stoechiometry_correction
from dementmutant.defaults import _default_dtype


def _dummy_water_potential_function(day):
    return 0.0


def _dummy_temperature_function(day):
    return 20.0


class SubstrateDegradationOperator:
    """
    SubstrateDegradationOperator class

    :class:`dementmutant.SubstrateDegradationOperator` stores structural information about
    substrate degradation reactions.
    It holds substrates and monomers names, as well
    as the boolean map linking substrates to the
    monomers they produce by degradation.
functions_module_path
    It is also meant to hold modulating parameters for the degradation
    process. For now the only such parameter is the lignin/cellulose
    correction factor.

    Notes :

        - kinetic parameters for the reactions themselves are enymatic
          properties and as such are stored in degradation enzymes objects.

    """
    def __init__(self, substrate_names, monomer_names):
        """
        Initiate :class:`dementmutant.SubstrateDegradationOperator`

        Parameters
        -----------
            substrate_names : list of str
                list of substrates names
            monomer_names : list of str
                list of monomer names

        """
        #: list of str
        self.substrate_names = substrate_names
        #: list of str
        self.monomer_names = monomer_names
        #: int
        self.n_substrates = len(self.substrate_names)
        #: int
        self.n_monomers = len(self.monomer_names)
        #: :class:`numpy.ndarray` bool
        self.map = np.zeros((self.n_substrates, self.n_monomers), dtype=np.bool_)
        #: :class:`numpy.ndarray` float
        self.map_t = np.zeros_like(self.map).astype(_default_dtype)
        #: float
        self._LCI_slope = 0.0

    def set_map_from_func(self, map_func):
        """
        Set substrate/monomer association map using a user provided
        function.

        Parameters
        ----------

        map_func : function
            A function (string, string) -> bool mapping a pair (subtrate_name, monomer_name)
            to a boolean indicating if the substrate produces the monomer.


        """
        for isub, sub in enumerate(self.substrate_names):
            for imon, mon in enumerate(self.monomer_names):
                self.map[isub, imon] = map_func(sub, mon)

        self.map_t = self.map.astype(_default_dtype)

    def set_LCI_slope(self, slope):
        """
        Set the value of the lignin/cellulose correction factor.

        Parameters
        ----------
            slope : float
                Slope for the LCI correction factor
        """
        self._LCI_slope = slope

    def get_LCI_slope(self):
        """
        Returns the value of the lignin/cellulose correction factor.

        Return
        ------
            float
                Value of the ligning/cellulose correction factor
        """
        return self._LCI_slope

    def get_map_df(self):
        """

        Returns the boolean subtrate/monomer association map
        as a pandas Dataframe.

        Return
        ------
            pandas.DataFrame
                A Dataframe of substrate/monomer association map

        """
        df = pd.DataFrame(
                data=self.map,
                index=self.substrate_names,
                columns=self.monomer_names,
                copy=True
                )
        return df

    def get_metadata_dict(self):
        direct_attrs = ['monomer_names', 'substrate_names', '_LCI_slope']
        np_attrs = ['map', ]
        res = {}
        for a in direct_attrs:
            res[a] = getattr(self, a)
        for a in np_attrs:
            res[a] = getattr(self, a).tolist()
        return res

    def set_parameters_from_metadata(self, pdict, locator, functions_module):
        direct_attrs = ['_LCI_slope', ]
        np_attrs = ['map', ]
        for a in direct_attrs:
            setattr(self, a, pdict[a])
        for a in np_attrs:
            setattr(self, a, np.array(pdict[a]))
        self.map_t = self.map.astype(_default_dtype)


class MonomerUptakeOperator:
    """
    MonomerUptakeOperator class

    This class holds structural data about monomer uptake reactions.
    It stores monomer and uptake transporters names, as well
    as the boolean association matrix linking them together
    It also stores a taxa/uptake transporter production efficiency
    matrix with allows to apply a multiplicative factor representing
    the relative efficiency of uptake transporters constitutive production

    """

    __microbial_mass_reference_elements = ['C', 'T']
    __default_mic_mass_ref_elem = 'C'

    def __init__(self, monomer_names, transporter_names, taxa_names):
        """
        Parameters
        ----------

            monomer_names : list of str
                List of monomer names

            transporter_names: list of str
                List of uptake transporters names

        """
        #: list of str
        self.monomer_names = monomer_names
        #: list of str
        self.transporter_names = transporter_names
        #: list of str
        self.taxa_names = taxa_names
        #: int
        self.n_monomers = len(self.monomer_names)
        #: int
        self.n_transporters = len(self.transporter_names)
        #: int
        self.n_taxa = len(self.taxa_names)

        #: :class:`numpy.ndarray` (bool)
        self.map = np.zeros((self.n_monomers, self.n_transporters), dtype=np.bool_)
        #: :class:`numpy.ndarray` (float)
        self.map_t = self.map.astype(_default_dtype)
        #: str in {'T','C'}
        self.mic_mass_ref_elem = self.__default_mic_mass_ref_elem

        #: :class `numpy.ndarray` (float)
        self.tax_prod_relative_efficiency = np.ones((self.n_taxa, self.n_transporters),
                                           dtype=_default_dtype)


        self.__transporter_indexes = {k: i for i, k in enumerate(self.transporter_names)}
        self.__monomer_indexes = {k: i for i, k in enumerate(self.monomer_names)}
        self.__taxa_indexes = {k: i for i, k in enumerate(self.taxa_names)}

    def get_transporter_index(self, transporter_name):
        return self.__transporter_indexes[transporter_name]

    def get_monomer_index(self, monomer_name):
        return self.__monomer_indexes[monomer_name]

    def get_taxon_index(self, taxon_name):
        return self.__taxa_indexes[taxon_name]

    def set_microbial_mass_reference_element(self, mic_ref_elem):
        """
        Set microbial mass reference element for uptake computation
        This can be either 'C' (C-based) or 'T' (total mass)

        Parameters
        ----------
            mic_ref_elem : str in {'C','T'}
                Reference microbial mass element 'C' (C-based) or 'T' (total mass)
        """

        if (mic_ref_elem in self.__microbial_mass_reference_elements):
            self.mic_mass_ref_elem = mic_ref_elem
        else:
            print('Allowed reference microbial biomass are {} - Aborting'.format(
                    self.__microbial_mass_reference_elements)
                  )
            assert(False)

    def get_microbial_mass_reference_element(self):
        """
            Return microbial mass reference element for uptake transporters computation
            This can be either 'C' (C-based) or 'T' (total mass)

        Return
        ------
            str in {'C','T'}
                Microbial mass element used as a basis for uptake transporters computation
                in monomer uptake reaction.
                This can be either 'C' (C-based) or 'T' (total mass based)
        """

        return self.mic_mass_ref_elem

    def get_microbe_references(self, mic_field):
        """
            Return microbial biomassfield array references used as
            a basis for computation of uptake transporters local concentration
            (C concentration or total mass C+N+P concentration)

        Parameters
        ----------

            mic_field : :class:`dementmutant.biomassfield.BiomassField`
                Microbial BiomassField object

        Return
        ------

            :class:`numpy.ndarray`
                Reference to :class:`numpy.ndarray` biomass buffer (mass or tmass)
            int
                element index in the bimoass array to for uptake transporters concentration
                computation

        """
        mic_ref = self.get_microbial_mass_reference_element()
        mic_ref_biomass = {
                'T': mic_field.tmass,
                'C': mic_field.mass,
                }[mic_ref]
        mic_ref_index = {
                'T': 0,
                'C': mic_field.get_atom_index('mass', 'C')
                }[mic_ref]
        return mic_ref_biomass, mic_ref_index

    def set_map_from_func(self, map_func):
        """

        Set the monomer/uptake transporter assocation using a user
        provided function (string, string) --> bool. Given a pair
        (monomer name, uptake transporter name), this function must return
        True if the uptake transporter acts as an enzymatic modulator
        in the monomer consumption by microbes

        Parameters
        ----------
            map_func : function (string, string) --> bool

        """
        for imon, mon in enumerate(self.monomer_names):
            for iupt, upt in enumerate(self.transporter_names):
                self.map[imon, iupt] = map_func(mon, upt)
        self.map_t = self.map.astype(_default_dtype)

    def get_map_df(self):
        """
        Returns the monomer/uptake transporter association matrix
        as a panda DataFrame

        Return
        ------
            :class:`panda.DataFrame`
                A pandas DataFrame storing the monomer/uptake transporter
                association matrix

        """
        df = pd.DataFrame(
                data=self.map,
                index=self.monomer_names,
                columns=self.transporter_names,
                copy=True
                )
        return df

    def get_taxa_relative_efficiency_df(self):
        df = pd.DataFrame(
            data = self.tax_prod_relative_efficiency,
            index = self.taxa_names,
            columns = self.transporter_names,
            copy = True
            )
        return df

    def get_metadata_dict(self):
        direct_attrs = ['monomer_names', 'transporter_names', 'taxa_names']
        np_attrs = ['map', 'tax_prod_relative_efficiency']
        res = {}
        for a in direct_attrs:
            res[a] = getattr(self, a)
        for a in np_attrs:
            res[a] = getattr(self, a).tolist()
        return res

    def set_parameters_from_metadata(self, pdict, locator, functions_module):
        direct_attrs = []
        np_attrs = ['map','tax_prod_relative_efficiency']
        for a in direct_attrs:
            setattr(self, a, pdict[a])
        for a in np_attrs:
            setattr(self, a, np.array(pdict[a]))
        self.map_t = self.map.astype(_default_dtype)


class LinearDecaySinkOperator:
    """
    LinearDecaySinkOperator class

    A lightweight class to describe  a linear decay process :math:`d_t F = -k F`

    Biomass losses are pooled in an internal biomass field for accounting.
    If an external target is provided, losses are also transferred to it.

    Class members:

        - source, target : biomassfield objects, target has same parameters as source
        - rate_func : a function yielding the decay rate either
            + as a scalar function of environnment
            + as a vector function (one per component) of environment

    """
    def __init__(self, source_field, external_target=None, target_component=None):
        #: :class:`dementmutant.biomassfield.BiomassField`
        self.source = source_field
        if (external_target is not None):
            delta_buffs = ['mass']
        else:
            delta_buffs = []
        #: :class:`dementmutant.biomassfield.BiomassField`
        self.target = BiomassField(source_field.grid_shape,
                                   source_field.names,
                                   biomass_dtype=source_field.get_biomass_dtype(),
                                   delta_buffers=delta_buffs
                                   )

        #: function
        self.rate_func = None
        #: str
        self.rate_func_name = ""
        #: :class:`dementmutant.biomassfield.BiomassField`
        self.external_target = external_target
        if (target_component is not None):
            assert(target_component in external_target.names)
        #: str
        self.target_component = target_component
        #: :class:`numpy.ndarray` float
        self._rates = np.zeros((source_field.n_components,),
                               dtype=source_field.get_biomass_dtype()
                               )
        #: bool
        self.rate_needs_source = False

    def set_rate_func(self, rate_func):
        """
        Sets the decay rate function

        Parameters
        ----------
            rate_func : function
                Decay rate function
        """
        allowed_params = ['env', 'source']
        sig_p = inspect.signature(rate_func).parameters
        for k in sig_p.keys():
            if k not in allowed_params:
                print('wrong parameter name in rate function signature')
                print('allowed parameters names {}'.format(allowed_params))
                self.rate_func = None
                return
        if (len(sig_p) > 2):
            print('Too many arguments in rate function signature')
            print('Maximal arguments are (env, source)')
            self.rate_func = None
            return
        self.rate_func = rate_func
        self.rate_func_name = rate_func.__name__

    def apply_decay(self, env):
        """
        Apply decay rate operator

        Parameters
        ----------
            env : dict
                Dictionnary of environnmental variables ('temp','psi')
        """
        if (self.rate_func is None):
            return
        rate_f_args = {}

        f_params = inspect.signature(self.rate_func).parameters
        if ('source' in f_params):
            rate_f_args['source'] = self.source
        if ('env' in f_params):
            rate_f_args['env'] = env
        self._rates[:] = self.rate_func(**rate_f_args)
        src = self.source.mass
        if (self.external_target is not None):
            targ = self.target.delta_mass
            targ[()] = 0.0
        else:
            targ = self.target.mass
        src, targ = _apply_decay(src, targ, self._rates)
        if (self.external_target is not None):
            self.target.mass += targ
            if (self.target_component is None):
                self.external_target.mass += self.target.delta_mass
            else:
                cid = self.external_target.get_component_index(self.target_component)
                self.external_target.mass[:, :, cid, :] += np.sum(self.target.delta_mass, axis=2)

        self.source.update_biomass_fields()
        self.target.update_biomass_fields()
        if (self.external_target is not None):
            self.external_target.update_biomass_fields()

        del(rate_f_args)

    def get_metadata_dict(self):
        direct_attrs = ['source', 'rate_func_name', 'external_target', 'target_component']
        res = {a: getattr(self, a) for a in direct_attrs}
        return res

    def set_parameters_from_metadata(self, pdict, locator, functions_module):
        self.rate_func_name = pdict['rate_func_name']
        self.rate_func = getattr(functions_module, self.rate_func_name)


def linear_decay_op_from_dict(pdict, locator, functions_module):
    """
    This function builds a :class:`dementmutant.ecosystem.LinearDecaySinkOperator` from
    saved parameters stored in a dictionnary
    (used for reloading state from json file)

    Parameters
    ----------
        pdict : dict
            dictionnary of parameters as reoaded from json files

        locator : :class:`dementmutant.ecosystem.BiomassFieldLocator`
            A instance of :class:`dementmutant.ecosystem.BiomassFieldLocator` used
            to resolve references to biomassfields

        function_module : python module
            module from which model functions are imported

    """
    src_name = locator._field_from_field_str(pdict['source'])
    source_field = locator.get_field(src_name)
    args = {'source_field': source_field}
    if (pdict['external_target'] is not None):
        ext_name = locator._field_from_field_str(pdict['external_target'])
        args['external_target'] = locator.get_field(ext_name)
    args['target_component'] = pdict['target_component']

    op = LinearDecaySinkOperator(**args)

    rate_func = getattr(functions_module, pdict['rate_func_name'])

    op.set_rate_func(rate_func)

    return op


class AutonomousSourceOperator:
    """

    Small class for adding time dependent external inputs
    to biomassfields

    Given a field :math:`F` and a function :math:`s(F,t)`, this operator
    is a first order euler scheme for the equation :math:`d_F = s(F,t)`

    """
    def __init__(self, target, source_func, parent_ecosystem):
        """
        Setup source operator

        Parameters
        ----------

        target : :class:`dementmutant.biomassfield.BiomassField`
            the biomass pool to which the source is applied

        source_func : function
            a function taking a biomass field and time index as arguments
            and returning the increment to be applied to the target

        parent_ecosystem : :class:`dementmuatant.ecosystem.Ecosystem`
            The `Ecosystem` object in which the source is defined.

        """
        assert(isinstance(target, BiomassField))
        #: :class:`dementmutant.biomassfield.BiomassField`
        self.target = target
        #: :class:`dementmutant.biomassfield.BiomassField`
        self.pool = BiomassField(
                                 self.target.grid_shape,
                                 self.target.names,
                                )
        #: function
        self.source_func = source_func
        #: str
        self.source_func_name = source_func.__name__

        assert(isinstance(parent_ecosystem, Ecosystem))
        #: :class:`dementmuatant.ecosystem.Ecosystem`
        self.ecosystem = parent_ecosystem

        #: bool
        self.requires_ecosystem = False
        if ('ecosystem' in inspect.signature(self.source_func).parameters):
            self.requires_ecosystem = True
        # bool
        self.requires_parameters = False
        if ('parameters' in inspect.signature(self.source_func).parameters):
            self.requires_parameters = True

        #: dict
        self.custom_parameters = {}

    def apply(self, itime):
        """

        Apply source increment at time index itime

        Parameters
        ----------
            itime : int
                time index

        """

        sf_args = (self.target,itime)
        sf_kwargs = {}
        if (self.requires_ecosystem):
            sf_kwargs['ecosystem'] = self.ecosystem
        if (self.requires_parameters):
            sf_kwargs['parameters'] = self.custom_parameters
        src = self.source_func(*sf_args,**sf_kwargs)
        #     src = self.source_func(self.target, itime,
        #                            ecosystem=self.ecosystem)
        # else:
        #     src = self.source_func(self.target, itime)

        self.target.mass += src
        self.pool.mass += src

        self.target.update_biomass_fields()

    def get_metadata_dict(self):
        res = {
                'source_func_name': self.source_func_name,
                'target': self.target,
                'custom_parameters':self.custom_parameters,
                }
        return res

    def set_parameters_from_metadata(self, pdict, locator, functions_module):
        if ('custom_parameters' in pdict.keys()):
            self.custom_parameters = pdict['custom_parameters'].copy()
        return

    def get_functions_set(self):
        return set([self.source_func, ])


def autotonomous_source_from_pdict(pdict, locator, functions_module, parent_ecosystem):
    """

    This function builds an autonomous source object from
    parameters stored in a dictionnary.
    (meant for reloading state from saved data)

    Parameters
    ----------

        pdict : dict
            dictionnary of parameters as reoaded from json files

        locator : :class:`dementmutant.ecosystem.BiomassFieldLocator`
            A instance of :class:`dementmutant.ecosystem.BiomassFieldLocator` used
            to resolve references to biomassfields

        functions_module : python module
            module from which model functions are imported

        parent_ecosystem : :class:`dementmuatant.ecosystem.Ecosystem`
            Ecosystem to which the source is attached

    Returns
    -------
        :class:`dementmutant.ecosystem.AutonomousSourceOperator`
            A :class:`dementmutant.ecosystem.AutonomousSourceOperator`
            set from saved parameters.

    """
    target = locator.get_field(locator._field_from_field_str(pdict['target']))
    source_func = getattr(functions_module, pdict['source_func_name'])
    res = AutonomousSourceOperator(target, source_func, parent_ecosystem)
    res.set_parameters_from_metadata(pdict, locator, functions_module)
    return res


class Ecosystem:
    """

    Ecosystem class : instantiate soil degradation model from DEMENT

    "monster" object storing
         - ecosystem biomass pools
         - microbial and biochemical parameters
         - methods for dynamical evolution through various processes

    """

    def __init__(self,
                 grid_shape,
                 substrate_names,
                 degradation_enzyme_names,
                 monomer_names,
                 uptake_transporter_names,
                 osmolyte_names,
                 taxa_names,
                 model_functions_module_name='model_functions',
                 model_functions_module_path=None,
                 ):
        """
        Parameters
        ----------
            grid_shape : 2-tuple of ints
                Shape of space grid
            substrate_names : list of str
                List of substrate names
            degradation_enzyme_names : list of str
                List of degradation enzymes names
            monomer_names : list of str
                List of monomer names
            uptake_transporter_names : list of str
                List of uptake transporter names
            osmolyte_names : list of str
                 List of osmolytes names
            taxa_names : list of str
                List of taxa names

            model_functions_module_name : str
                Name of the python module from which to import model functions.
                CAVEAT : When running multiple instances with different model
                functions from the same python process, ensure that there is no module name collision, or only
                the first loaded module will be effectively used.

            model_functions_module_path : str or None
                Path of source file for model functions. If provided, its content
                will be loaded as a module named from model_functions_module_name.
                If absent, the module is assumed importable from sys.path.

        """
        #: 2-tuple of ints
        self.grid_shape = grid_shape
        #: :class:`dementmutant.substrate.Substrate`
        self.substrates = Substrate(self.grid_shape, substrate_names)
        #: :class:`dementmutant.monomer.Monomer`
        self.monomers = Monomer(self.grid_shape, monomer_names)
        #: :class:`dementmutant.ecosystem.SubstrateDegradationOperator`
        self.substrate_degradation_op = SubstrateDegradationOperator(substrate_names,
                                                                     monomer_names,
                                                                     )
        #: :class:`dementmutant.enzyme.DegradEnzyme`
        self.degradation_enzymes = DegradEnzyme(
                                        self.grid_shape,
                                        degradation_enzyme_names,
                                        substrate_names,
                                        )
        #: :class:`dementmutant.ecosystem.MonomerUptakeOperator`
        self.monomer_uptake_op = MonomerUptakeOperator(monomer_names, uptake_transporter_names, taxa_names)
        #: :class:`dementmutant.enzyme.UptakeTransporter`
        self.uptake_transporters = UptakeTransporter(uptake_transporter_names, monomer_names)
        #: :class:`dementmutant.microbe.Microbe`
        self.microbes = Microbe(self.grid_shape, taxa_names, delta_buffers=['mass'])
        ###self.microbes = Microbe(self.grid_shape, taxa_names, delta_buffers=['mass'])

        self.microbes.add_metabolite("Uptake_Transporters", uptake_transporter_names)
        self.microbes.add_metabolite("Degradation_Enzymes", degradation_enzyme_names)
        self.microbes.add_metabolite("Osmolytes", osmolyte_names)
        # add pseudo-metabolite for Respiration/Growth
        self.microbes.add_metabolite("Respiration_Growth", ["RespGrowth", ])
        #: dict of :class:`dementmutant.ecosystem.LinearDecaySinkOperator`
        self.linear_decay_operators = {}
        #: dict of :class:`dementmutant.ecosystem.AutonomousSourceOperator`
        self.source_operators = {}
        #: :class:`dementmutant.ecosystem.BiomassFieldLocator`
        self.field_locator = BiomassFieldLocator(self)
        #
        self.update_field_locator()

        #: function
        self._environment_function = None
        #: str
        self._environment_function_name = ''

        #: str
        self.functions_module_name = model_functions_module_name

        #: str or None
        self.functions_module_path = model_functions_module_path
        #: python module
        # self.functions_module = __import__(self.functions_module_name)
        self.functions_module = None

        if (self.functions_module_path is None):
            self.functions_module = importlib.import_module(self.functions_module_name)
        else:
            spec = importlib.util.spec_from_file_location(self.functions_module_name,
                                                          self.functions_module_path,
                                                          )
            self.functions_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.functions_module)

        #: dict
        self.custom_parameters = {}

    def get_metadata_dict(self):
        res = {}
        res['grid_shape'] = self.grid_shape
        dlist = ['substrates',
                 'monomers',
                 'degradation_enzymes',
                 'uptake_transporters',
                 'substrate_degradation_op',
                 'monomer_uptake_op',
                 'microbes',
                 'field_locator',
                 ]

        for a in dlist:
            res[a] = getattr(self, a).get_metadata_dict()

        for a in ['linear_decay_operators', 'source_operators']:
            res[a] = {k: d.get_metadata_dict() for k, d in getattr(self, a).items()}

        res['_environment_function_name'] = self._environment_function_name

        if (hasattr(self, 'custom_parameters')):
            res['custom_parameters'] = self.custom_parameters
        return res

    def set_parameters_from_metadata(self, pdict):
        dlist = ['substrates',
                 'monomers',
                 'degradation_enzymes',
                 'uptake_transporters',
                 'substrate_degradation_op',
                 'monomer_uptake_op',
                 'microbes',
                 ]
        for a in dlist:
            o = getattr(self, a)
            o.set_parameters_from_metadata(pdict[a],
                                           self.field_locator,
                                           self.functions_module
                                           )
            self.field_locator.update()

        for k, d in pdict['linear_decay_operators'].items():
            self.linear_decay_operators[k] = linear_decay_op_from_dict(d,
                                                                       self.field_locator,
                                                                       self.functions_module,
                                                                       )
            self.field_locator.update()

        for k, d in pdict['source_operators'].items():
            self.source_operators[k] = autotonomous_source_from_pdict(d,
                                                                      self.field_locator,
                                                                      self.functions_module,
                                                                      self,
                                                                      )
            self.field_locator.update()

        if (len(pdict['_environment_function_name']) > 0):
            func = getattr(self.functions_module, pdict['_environment_function_name'])
            self.set_environment_function(func)

        self.field_locator.set_parameters_from_metadata(pdict['field_locator'])

        if ('custom_parameters' in pdict.keys()):
            if isinstance(pdict['custom_parameters'],dict):
                self.custom_parameters = pdict['custom_parameters'].copy()
#    def get_functions_set(self):
#        res = [self._environment_function, ]
#        res += list(self.microbes.get_functions_set())
#        for op in self.source_operators.values():
#            res += list(op.get_functions_set())
#        return set(res)

    def save_parameters_to_json(self, filename):
        """
        Save lightweight structural data and parameters
        to a json file.
        Save data can be used to regenerate the object
        state, except for biomass fields.

        Parameters
        ----------
            filename : str
                Path of json file

        """
        # get metadata dictionnary
        sdict_raw = self.get_metadata_dict()
        #
        # convert biomassfield pointers to string descriptors
        s_dict = self.field_locator.field_to_str(sdict_raw)
        with open(filename, 'w') as f:
            json.dump(s_dict, f, indent=4)

    def save_functions_module(self, save_dir):
        """
            Save model function module to file for reloading/reuse
            For now this is a simple copy of the static file initialy provided
        """
        # TODO : implement function collection and dynamical rewrite of module

        init_file = self.functions_module.__file__
        export_file = '{}/model_functions.py'.format(save_dir)
        exp_exist = os.path.exists(export_file)
        if exp_exist:
            if  not os.path.samefile(init_file, export_file):
                shutil.copy2(init_file, export_file)
        else:
            shutil.copy2(init_file, export_file)



    def set_environment_function(self, func):
        """

        Define the environnment function yielding environmental parameters
        (temperature, water potential) as a function of the time index.

        Parameters
        ----------

            func : function (int) --> {'temp':float, 'psi': float}
                A function mapping the time index and returning a dictionnary
                of environmental parameters

        """
        self._environment_function = func
        self._environment_function_name = func.__name__

    def get_environment(self, itime):
        """
            Returns environnmental parameters values from time index
            Use either environment function if it was set or output
            default dummy values if not.

        Parameters
        ----------
            itime : int
                Time index

        """
        if (self._environment_function is not None):
            res = self._environment_function(itime)
        else:
            print('Using dummy environment')
            psi = _dummy_water_potential_function(itime)
            temp = _dummy_temperature_function(itime)
            res = {'temp': temp, 'psi': psi}
        return res

    def apply_monomer_diffusion(self, itime):
        """
        Apply effect of monomer diffusion in space
        This simple assumes infinite diffusion coefficient, and
        null gradient boundary conditions : all fields are replaced
        by their space average whatever value the time step has.
        """

        means = self.monomers.get_space_average()
        self.monomers.mass[:, :, :, :] = means[np.newaxis, np.newaxis, :, :]

        self.monomers.update_biomass_fields()

        del(means)

    def apply_subtrate_degradation(self, itime):
        """

        Apply effect of substrate degradation to monomers, catalysed
        by substrates degradation enzymes.


        :todo:
            - make ligning/cellulose correction factor optional

        """
        # get current environmental conditions
        env = self.get_environment(itime)
        # compute reaction kinetic parameters
        Vmax, Km = self.degradation_enzymes.get_kinetic_parameters(env)
        #
        # Apply lignin/cellulose correction
        lignin_indexes = (
                          self.substrates.get_component_index('Lignin'),
                          0
                          )
        cellulose_indexes = (
                             self.substrates.get_component_index('Cellulose'),
                             self.substrates.get_atom_index('mass', 'C')
                             )

        enz_C_index = self.degradation_enzymes.get_atom_index('mass', 'C')

        LCI_slope = self.substrate_degradation_op.get_LCI_slope()

        self.substrates.mass, self.monomers.mass = _sub_degrad_decay_rate(
                    self.substrates.mass, self.substrates.tmass, self.substrates.ratios,
                    self.degradation_enzymes.mass, enz_C_index,
                    self.monomers.mass,
                    self.substrate_degradation_op.map_t,
                    Vmax, Km,
                    lignin_indexes, cellulose_indexes,
                    LCI_slope,
                )

        self.substrates.update_biomass_fields()
        self.monomers.update_biomass_fields()

    def apply_monomer_uptake_and_inducible_metabolism(self, itime):
        """
            Apply
                - monomer uptake by microbes
                - inducible metabolic processes (computed from uptake)
        """

        self.apply_monomer_uptake_only(itime)

        self.apply_inducible_only(itime)

    def apply_monomer_uptake_only(self, itime):
        """
            Compute only monomer uptake and store it in microbes
            secondary buffer (delta_mass)
        """
        # get current environnmental conditions
        env = self.get_environment(itime)
        # compute reaction kinetic parameters
        Vmax, Km = self.uptake_transporters.get_kinetic_parameters(env)
        # apply monomer decay / microbe feeding
        self.microbes.delta_mass[()] = 0  # zero out delta mass
        # microbial mass time derivative is computed and stored for further use
        taxupt_C_prod = (
            self.microbes.get_metabolite('Uptake_Transporters').ref_cost['Consti']
            * self.monomer_uptake_op.tax_prod_relative_efficiency
            )
        mic_ref, mic_ref_index = self.monomer_uptake_op.get_microbe_references(self.microbes)
        self.monomers.mass, self.microbes.delta_mass = _mon_uptake_decay_rate(
                self.monomers.mass,
                self.monomers.ratios,
                self.monomers.tmass,
                self.microbes.delta_mass,
                mic_ref,
                mic_ref_index,
                self.monomer_uptake_op.map_t,
                taxupt_C_prod,
                Vmax,
                Km
                )
        # update monomer total masses and ratios
        self.monomers.update_biomass_fields()


    def apply_maxuptake_to_microbes(self,itime):
        """
        For debggin purposes -
        Apply full monomer uptake to microbes without
        inducible consumption
        Parameters
        ----------
        itime : int
            timestep.

        Returns
        -------
        None.

        """
        self.microbes.mass = _apply_delta_sum(self.microbes.mass,
                                              self.microbes.delta_mass
                                              )
        #
        self.microbes.update_biomass_fields()

    def apply_inducible_only(self, itime):
        """
        Computes uptake fraction taken out from uptake stored
        in microbes delta_mass by inducible metabolic processes
        Update microbial biomass with the net result
        Update various inducible metabolims biomass pools for accounting.
        """
        #
        self.apply_metabolic_processes('Induci', itime)
        #
        self.microbes.mass = _apply_delta_sum(self.microbes.mass,
                                              self.microbes.delta_mass
                                              )
        #
        self.microbes.update_biomass_fields()
        #

    def apply_metabolic_processes(self, process, itime):
        """
            Apply a set of metabolic processes of a given type
            Inducible processes transfer biomass from microbes delta_mass
            ( which should contain already computed monomer uptake biomass)
            Constitutive processes transfer biomass from microbes mass.

        Parameters
        ----------
            process :str
                Metabolic process type ('Induci' or 'Consti')
            itime : int
                time index
        """
        env = self.get_environment(itime)

        targ_src_costs, src_costs, targ_ranks = (
                    self.microbes.get_metabolic_processes_operands(process, env)
                                                )
        self.microbes.clear_metabolic_duplicate_targets_buffers(process)
        targets = self.microbes.get_metabolic_processes_targets(process)
        tax_c_index = self.microbes.get_atom_index('mass', 'C')

        source = {
                "Consti": self.microbes.mass,
                "Induci": self.microbes.delta_mass,
                }[process]

        source, targets = _metabolic_process_ranked(
                    source, tax_c_index, src_costs,
                    targets, targ_src_costs, targ_ranks
                    )

        self.microbes.transfer_metabolic_duplicate_targets(process)

        self.microbes.sync_targets(process)

        self.microbes.sync_external_targets(process)

        if (process == 'Consti'):
            self.microbes.update_biomass_fields()

    def apply_quantized_birth_mut(self,itime,rng):
        """
        WIP : birth/mutation process base on local mass quantization.
        Parameters
        ----------
        itime : integer
            Time index
        rng : np.random.generator
            Random generator for mutation process

        Returns
        -------
        None.

        """

        if (not self.microbes.is_quantized):
            print('Cannot compute birth on non quantized field')
            return

        self.microbes.update_quantization(constraint='up')

        nbirth = np.sum(self.microbes.delta_quanta)
        nmut = 0

        if (not self.microbes.mutation_op.is_active):
            self.microbes.mutation_op.last_run_stats = {
                                                   'nbirth': nbirth,
                                                   'nmut': nmut,
                                                   }
            return

        self.microbes.mutation_op.compute_mutation_probas()

        mass = self.microbes.mass
        dmass = self.microbes.delta_mass
        quant = self.microbes.quanta
        dquant = self.microbes.delta_quanta

        nr,nc,ntax,na = mass.shape

        # nbirth =
        maxnbirth = int(np.max(dquant))

        draws = np.zeros(quant.shape[:-1]+(maxnbirth,),dtype=int)
        drsize = draws[:,:,0,:].shape
        for itax in range(ntax):
            draws[:,:,itax,:] = rng.choice(
                ntax,
                size= drsize,
                p = self.microbes.mutation_op.mutation_probas[itax,:],
                replace=True,
                )
        dmass[()] = 0
        #
        mutop = {
            'one' : _apply_quantized_birth_mutation,
            'two' : _apply_quantized_birth_mutation2,
            }[self.microbes.mutation_op.mutation_n_daughters]
        #
        mass,dmass,quant,nmut = mutop(mass,
                                      dmass,
                                      quant,
                                      dquant,
                                      draws)


        self.microbes.mutation_op.last_run_stats = {
                                                    'nbirth': nbirth,
                                                    'nmut': nmut,
                                                    }

        self.microbes.mass = _apply_delta_sum(self.microbes.mass,
                                              self.microbes.delta_mass
                                              )
        self.microbes.update_biomass_fields()

        del(draws)

    def apply_stoechiometry_only(self,itime):
        """
        Apply stoechiometric correction to microbial mass
        to account for non-modeled metabolic processes and
        avoid unrealistic microbial stoechiometry

        Parameters
        ----------
        itime : int
            Time index.

        Returns
        -------
        None.

        """

        if (not self.microbes.stoechio_balance_recycler.is_active):
            return

        mass = self.microbes.mass
        rmass = self.microbes.ratios

        stoech_quotas = self.microbes.get_taxa_stoechiometry_quotas('min')

        recycled_mass = self.microbes.stoechio_balance_recycler.delta_mass[:, :, 0, :]
        recycled_mass[()] = 0.0

        mass, rmass, recycled_mass = _apply_stoechiometry_correction(mass,
                                                                     rmass,
                                                                     recycled_mass,
                                                                     stoech_quotas
                                                                     )

        self.microbes.stoechio_balance_recycler.apply_transfers()
        self.microbes.update_biomass_fields()


    def apply_mortality_only(self, itime, rng):
        """
            Apply various mortality processes

                - deterministic starvation by insuffient biomass local content.
                - stochastic death modulated by environmental conditions (drought)
        """

        if (not self.microbes.mortality_op.is_active):
            return


        env = self.get_environment(itime)
        death_proba = self.microbes.mortality_op.get_mortality_proba(env)




        mass = self.microbes.mass
        #
        dead_mass = self.microbes.mortality_op.delta_mass[:, :, 0, :]
        dead_mass[()] = 0.0
        #
        draw_shape = self.microbes.mass.shape[:-1]
        if (self.microbes.is_quantized):
            maxq = int(np.max(self.microbes.quanta))
            draw_shape = self.microbes.mass.shape[:-1] + (maxq,)

        draws = rng.uniform(size=draw_shape).astype(_default_dtype)

        if (self.microbes.is_quantized):
            quant = self.microbes.quanta
            mass, quant, dead_mass = (
                    _apply_quantized_mortality(
                                     mass, quant, dead_mass,
                                     self.microbes.mortality_op.starvation_thresholds,
                                     death_proba,
                                     draws,
                                     )
                    )
        else:
            mass,dead_mass = (
                    _apply_mortality_stoech(
                                     mass, dead_mass,
                                     self.microbes.mortality_op.starvation_thresholds,
                                     death_proba,
                                     draws,
                                     )
                                                    )
        self.microbes.mortality_op.apply_transfer()
        self.microbes.update_biomass_fields()

        del(draws)

        return




    def apply_mortality_stoech(self, itime, rng):
        """
            Apply various mortality processes

                - deterministic starvation by insuffient biomass local content.
                - stochastic death modulated by environmental conditions (drought)
                - biomass recycling due to deviation from nominal stoechiometry
        """

        # ! TODO finer grained dswitches for recycling / stochastic death
        if (not self.microbes.mortality_op.is_active):
            return


        env = self.get_environment(itime)
        death_proba = self.microbes.mortality_op.get_mortality_proba(env)
        stoech_quotas = self.microbes.get_taxa_stoechiometry_quotas('min')




        mass = self.microbes.mass
        rmass = self.microbes.ratios
        #
        dead_mass = self.microbes.mortality_op.delta_mass[:, :, 0, :]
        dead_mass[()] = 0.0
        #
        recycled_mass = self.microbes.stoechio_balance_recycler.delta_mass[:, :, 0, :]
        recycled_mass[()] = 0.0

        draw_shape = self.microbes.mass.shape[:-1]
        if (self.microbes.is_quantized):
            maxq = int(np.max(self.microbes.quanta))
            draw_shape = self.microbes.mass.shape[:-1] + (maxq,)

        draws = rng.uniform(size=draw_shape).astype(_default_dtype)

        if (self.microbes.is_quantized):
            quant = self.microbes.quanta
            mass, rmass, quant, dead_mass, recycled_mass = (
                    _apply_quantized_mortality_stoech(
                                     mass, rmass, dead_mass, recycled_mass,
                                     quant,
                                     self.microbes.mortality_op.starvation_thresholds,
                                     death_proba,
                                     draws,
                                     stoech_quotas,
                                     )
                    )
        else:
            mass, rmass, dead_mass, recycled_mass = (
                    _apply_mortality_stoech(
                                     mass, rmass, dead_mass, recycled_mass,
                                     self.microbes.mortality_op.starvation_thresholds,
                                     death_proba,
                                     draws,
                                     stoech_quotas,
                                     )
                                                    )
        self.microbes.mortality_op.apply_transfer()
        self.microbes.stoechio_balance_recycler.apply_transfers()

        self.microbes.update_biomass_fields()

        del(draws)
        return

    def apply_linear_decay_ops(self, itime):
        """
        Apply global linear decay operators
        """
        env = self.get_environment(itime)
        for k, op in self.linear_decay_operators.items():
            op.apply_decay(env)

    def set_linear_decay_operator(self, name, source, rate_func,
                                  external_target=None, target_component=None
                                  ):
        """
        Setup a linear decay operator, modeling biomass losses

        Parameters
        ----------

            name : str
                User defined name for the loss process

            source : :class:`dementmutant.biomassfield.BiomassField`
                A biomass field subject to losses

            rate_func : function
                User provided function yielding the decay rate

            external_target (optional) : :class:`dementmutant.biomassfield.BiomassField`
                A biomass field to which masses losses are transfered

            target_component : str
                Name of the component of the external target to which biomass
                is transfered.

        """
        op = LinearDecaySinkOperator(source, external_target, target_component)
        op.set_rate_func(rate_func)
        self.linear_decay_operators[name] = op
        self.update_field_locator()

    def apply_quantized_dispersal_old(self, itime, disp_rng):
        """
        Quantized spatial dispersion
        Randomly transport microbial biomass quanta
        CAVEAT : for now the same process is used for both
        bacteria and fungi, ie mass is displaced with a given
        shift probability distribution for each quantum ( which can be different
        for each species). The quantum value can also be modulated per taxon.
        Parameters
        ----------
        itime : int
            Time index.
        disp_rng : numpy random generator
            Generator used to draw dispersal vector.

        Returns
        -------
        None.

        """
        if (not self.microbes.is_quantized):
            return
        if not self.microbes.dispersal_op.is_active:
            return



        mass = self.microbes.mass
        dmass = self.microbes.delta_mass
        quant = self.microbes.quanta
        dquant = self.microbes.delta_quanta

        dmass[()] = 0.0
        dquant[()] = 0.0

        for mic_type, filt in self.microbes.type_filters.items():
            if (self.microbes.dispersal_op.types[mic_type] is None):
                continue
            if (not np.any(filt)):
                continue
            ndisp = int(np.sum(self.microbes.quanta[:,:,filt,0]))
            if (ndisp == 0):
                continue
            disp_par = self.microbes.dispersal_op.get_dispersal_prob_distrib(mic_type)
            shifts = np.zeros((ndisp,len(disp_par)), dtype=int)
            for i, kdim in enumerate(disp_par.keys()):
                shifts[:,i] = disp_rng.choice(disp_par[kdim]['vals'],
                                        size=ndisp,
                                        replace=True,
                                        p=disp_par[kdim]['probas']
                                        )
            # WIP
            # TODO design/implement fungi behavior ?
            # step 1 : move mass and quant to delta_mass delta_quanta

            mass,dmass,quant,dquant,nmov = _move_quanta_mass_to_delta_filtered(
                mass,dmass,quant,dquant,
                filt, shifts
                )

            del(shifts)

        mass += dmass
        quant += dquant

        self.microbes.mass = mass
        self.microbes.delta_mass = dmass
        self.microbes.quanta = quant
        self.microbes.delta_quanta = dquant

        self.microbes.update_biomass_fields()


    def apply_quantized_dispersal(self, itime, disp_rng):
        """
        Quantized spatial and genetic dispersion
        Randomly transport microbial biomass quanta
        Optionnaly Randomly change taxa indexes for taxa moving through domain boundaries
        (this mimics fluxes from neighbouring patches with the assumtion that
         overall population is similar (same number of individuals , but with
        a possibly different composition )
        CAVEAT : for now the same process is used for both
        bacteria and fungi, ie mass is displaced with a given
        shift probability distribution for each quantum ( which can be different
        for each species). The quantum value can also be modulated per taxon.
        Parameters
        ----------
        itime : int
            Time index.
        disp_rng : numpy random generator
            Generator used to draw dispersal vector.

        Returns
        -------
        None.

        """
        if (not self.microbes.is_quantized):
            return
        if not self.microbes.dispersal_op.is_active:
            return



        mass = self.microbes.mass
        dmass = self.microbes.delta_mass
        quant = self.microbes.quanta
        dquant = self.microbes.delta_quanta

        dmass[()] = 0.0
        dquant[()] = 0.0

        for mic_type, filt in self.microbes.type_filters.items():
            if (self.microbes.dispersal_op.types[mic_type] is None):
                continue
            if (not np.any(filt)):
                continue
            ntax_filt =np.sum(filt.astype(int))
            tax_ids = np.indices((filt.size,))[0]
            tax_filt_ids = tax_ids[filt]
            ndisp_per_tax = np.sum(self.microbes.quanta[:,:,:,0],axis=(0,1)).astype(int)
            ndisp = int(np.sum(ndisp_per_tax))
            if (ndisp == 0):
                continue
            disp_par = self.microbes.dispersal_op.get_dispersal_prob_distrib(mic_type)
            shifts = np.zeros((ndisp,len(disp_par)), dtype=int)
            for i, kdim in enumerate(disp_par.keys()):
                shifts[:,i] = disp_rng.choice(disp_par[kdim]['vals'],
                                        size=ndisp,
                                        replace=True,
                                        p=disp_par[kdim]['probas']
                                        )
            # WIP
            # TODO design/implement fungi behavior ?
            nmov,ncross,nchange = 0,0,0
            if (self.microbes.dispersal_op.types[mic_type] == 'quantized_bdflux'):
                ndisp_per_tax_filtered = ndisp_per_tax[filt]
                itaxup = np.cumsum(ndisp_per_tax_filtered)
                itaxlow = itaxup-ndisp_per_tax_filtered
                tax_shifts = -np.ones((ndisp,),dtype=int)
                for itaxfilt,itax in enumerate(tax_filt_ids):
                    ilow,iup = itaxlow[itaxfilt], itaxup[itaxfilt]
                    dr_size = ndisp_per_tax_filtered[itaxfilt]
                    tax_shifts[ilow:iup] = disp_rng.choice(
                        tax_filt_ids,
                        size=dr_size,
                        replace=True,
                        )

                # assert(np.min(tax_shifts) > -1)
                mass,dmass,quant,dquant,nmov, ncross, nchange = _move_quanta_mass_to_delta_filtered_taxflux(
                    mass,dmass,quant,dquant,
                    filt, shifts, tax_shifts
                    )
                del(tax_shifts)

            if (self.microbes.dispersal_op.types[mic_type] == 'quantized'):
                mass,dmass,quant,dquant,nmov = _move_quanta_mass_to_delta_filtered(
                    mass,dmass,quant,dquant,
                    filt, shifts
                    )
                ncross = 0
                nchange = 0
            self.microbes.dispersal_op.last_run_stats[mic_type] = {'nmov':nmov,'ncross':ncross,'nchange':nchange}
            del(shifts)

        # print('{} Ndisp {} Nmove {} Ncross {} Nchange {}'.format(
        #     self.microbes.dispersal_op.types[mic_type],
        #     ndisp,nmov,ncross,nchange), flush=True)

        # TODO collect ndisp,nmow,ncross, nchange somewhere to allow access
        # for diagnostics

        mass += dmass
        quant += dquant

        self.microbes.mass = mass
        self.microbes.delta_mass = dmass
        self.microbes.quanta = quant
        self.microbes.delta_quanta = dquant

        self.microbes.update_biomass_fields()



    def apply_dispersal(self, itime, disp_rng, mut_rng):
        """
            Compute microbial mass subject to dispersal
            and dispersal spatial shifts.
            Move dispersed mass across the grid
            If mutations are activated compute taxa mutation
            and apply inter-taxa movements.
        """
        if not self.microbes.dispersal_op.is_active:
            return

        # debug
        # init_check_mass = np.sum(self.microbes.mass)
        # init_check_quant = np.sum(self.microbes.quanta)

        if (self.microbes.is_quantized):
            self.apply_quantized_dispersal(itime, disp_rng)
        else:
            self.apply_dispersal_phase1(itime, disp_rng, mut_rng)
            self.apply_dispersal_phase2(itime, disp_rng, mut_rng)

        # self.apply_dispersal_phase1(itime, disp_rng, mut_rng)
        # self.apply_dispersal_phase2(itime, disp_rng, mut_rng)
        # end_check_mass = np.sum(self.microbes.mass)
        # end_check_quant = np.sum(self.microbes.quanta)

        # print('BBBBBBB mass {} {} {} '.format(init_check_mass,end_check_mass,end_check_mass-init_check_mass))
        # print('BBBBBBB quant {} {} {} '.format(init_check_quant,end_check_quant,end_check_quant-init_check_quant))


    def apply_dispersal_phase1(self, itime, disp_rng, mut_rng):
        """
        Dispersal operator phase 1:

        Compute excess microbial mass subject to dispersal and
        store it in microbes.delta_mass.

        Draw stochastisc spatial shifts per direction for
        excess biomass movement on the spatial grid.


        """

        if not self.microbes.dispersal_op.is_active:
            return


        thresh = self.microbes.dispersal_op.saturation_thresholds
        #
        self.microbes.dispersal_op.shifts[()] = 0
        self.microbes.delta_mass[()] = 0.0
        #
        for mic_type, filt in self.microbes.type_filters.items():
            if (self.microbes.dispersal_op.types[mic_type] is None):
                continue
            if (not np.any(filt)):
                continue

            if (self.microbes.dispersal_op.types[mic_type] == 'local_ratio'):
                rat = self.microbes.dispersal_op.biomass_ratios[mic_type]
                fdisp = self.microbes.dispersal_op.dispersion_filter
                fdisp[()] = False

                self.microbes.mass, self.microbes.delta_mass, fdisp, ndisp = (
                    _mass_ratio_removal(
                            self.microbes.mass,
                            self.microbes.delta_mass,
                            fdisp,
                            filt,
                            thresh,
                            rat
                            )
                    )
                print('ndisp {}'.format(ndisp))

            if (self.microbes.dispersal_op.types[mic_type] == 'non_local_sharing'):
                fdisp = self.microbes.dispersal_op.dispersion_filter
                fdisp[()] = False

                self.microbes.mass, self.microbes.delta_mass, fdisp, ndisp = (
                    _mass_nonlocal_sharing_removal(
                            self.microbes.mass,
                            self.microbes.delta_mass,
                            fdisp,
                            filt,
                            thresh,
                            )
                    )

            disp_par = self.microbes.dispersal_op.get_dispersal_prob_distrib(mic_type)
            for i, kdim in enumerate(disp_par.keys()):
                draws = disp_rng.choice(disp_par[kdim]['vals'],
                                        size=ndisp,
                                        replace=True,
                                        p=disp_par[kdim]['probas']
                                        )

                self.microbes.dispersal_op.shifts = (
                        _set_shifts_filtered(self.microbes.dispersal_op.shifts,
                                             fdisp,
                                             draws,
                                             i
                                             )
                        )
                del(draws)

    def apply_dispersal_phase2(self, itime, disp_rng, mut_rng):
        """
        Dispersal phase 2

        If mutations are present, compute mutation probability
        distribution and draw taxon->taxon movement.

        Move excess biomass across-grid and perform taxon
        displacement as well if mutations are activated

        :todo: move mutation drawing to phase 1

        """

        if not self.microbes.dispersal_op.is_active:
            return

        if (self.microbes.mutation_op.is_active and not self.microbes.is_quantized):
            self.microbes.mutation_op.compute_mutation_probas()
            self.microbes.mutation_op.compute_taxa_shifts(mut_rng)
            self.microbes.mass = (
                    _move_delta_mass_withmut(
                                             self.microbes.mass,
                                             self.microbes.delta_mass,
                                             self.microbes.dispersal_op.shifts,
                                             self.microbes.mutation_op.shifts,
                                             )
                                  )
        else:
            self.microbes.mass = _move_delta_mass(self.microbes.mass,
                                                  self.microbes.delta_mass,
                                                  self.microbes.dispersal_op.shifts
                                                  )
        self.microbes.update_biomass_fields()

    def apply_external_sources(self, itime):
        """

            Apply the effect of all sources operators sequentially
            if any are defined.

        """
        for k, src in self.source_operators.items():
            src.apply(itime)

    def set_external_source(self, name, target_field, source_func):
        """
            Add a new source operator

        Parameters
        ----------
            name : str
                User defined name for the source - beware if the name has
                alreay been used the new source operator will overwrite the previous one.

            target_field : :class:`dementmutant.biomassfield.BiomassField`
                The biomass field on which the source operator will act

            source_func : func
                A function taking a biomass field and time index as arguments
                and returning the increment to be applied to the target.

        """
        self.source_operators[name] = AutonomousSourceOperator(target_field, source_func, self)
        self.update_field_locator()

    def get_field_locator(self):
        """
        Ecosystem field locator reference

        Returns
        -------
            :class: `dementmutant.ecosystem.BiomassFieldLocator`
        """
        return self.field_locator

    def update_field_locator(self):
        """
            Recursively scans members for fields and register them
            it in the field locator with a hierarchical naming scheme
            and metadata tags

            As some processes transfer mass to preexistent biomass fields,
            fields considered as primary pools are tagged as 'pool', while
            duplicate ones are tagged as 'transient', to avoid double countings
            External sources are tagged as 'input' as well.

            Formally, pools are nodes of the mass flow graph (whose edges are
            biomass transfer through processes). Transients correspond to cases
            when the flow is duplicated to two nodes.

            Those tags ease the automatic verification of mass conservation
            (ie the sum of pools minus the sum of input over the whole
            grid is a constant in time)

        :todos:

            - clarify biomass flow graph
            - add utility to plot process flow graph

        """
        for a in dir(self):
            o = getattr(self, a)
            if (isinstance(o, BiomassField)):
                self.field_locator.register(a, o, tags=['pool', a])
        for metname, met in self.microbes.metabolites.items():
            for proc, d in met.targets.items():
                for targ, f in d.items():
                    if (not met.active_pathway[proc][targ]):
                        continue
                    fname = '{}_{}_{}'.format(metname, proc, targ)
                    tags = [proc, targ, metname]
                    if (met.external_targets[proc][targ] is None):
                        tags += ['pool', ]
                    else:
                        tags += ['transient', ]
                    self.field_locator.register(fname, f, tags=tags)

        for k, d in self.linear_decay_operators.items():
            if (d.external_target is None):
                self.field_locator.register(k, d.target, tags=[k, 'pool'])
            else:
                self.field_locator.register(k, d.target, tags=[k, 'transient'])

        for a in dir(self.microbes):
            o = getattr(self.microbes, a)
            cinstances = [StoechiometricBalanceRecycler, MicrobialMortality]
            if (any([isinstance(o, c) for c in cinstances])):
                self.field_locator.register(a, o, tags=[a, 'transient'])
        for k, d in self.source_operators.items():
            self.field_locator.register(k, d.pool, tags=[k, 'pool', 'input'])

    def check_biomass_finiteness(self, tag=''):
        """
        Scan biomassfields and, check that all values are finite
        and print out report

        Parameters
        ----------
            tag : str
                Arbitrary string that will be added to messages if one
                wants to perform several checks and discriminate between
                reports.
        """
        print('-'*40)
        for t in ['microbes', 'substrates', 'monomers', 'degradation_enzymes']:
            o = getattr(self, t)
            chk = o.check_finiteness()
            print('{} {} {}'.format(tag, t, chk))
        t = 'recycle'
        o = self.microbes.stoechio_balance_recycler
        chk = o.check_finiteness()
        print('{} {} {}'.format(tag, t, chk))

    def get_diag_collector(self):
        """
        Builds a :class:`dementmutant.ecosystem.DiagnosticCollector` object
        for this ecosystem.
        This simply calls the constructor and sets the field
        locator object to the one of the ecosystem, so that
        the diagnostic collector has references to all biomassfields
        data in the ecosystem.

        Return
        ------
            :class:`dementmutant.ecosystem.DiagnosticCollector`
                A :class:`dementmutant.ecosystem.DiagnosticCollector` instance
        """
        diagcollector = DiagnosticCollector(self)
        return diagcollector

    def get_taxon_monomer_maps(self):
        """
        Builds and returns association maps of taxa with monomers
        both as consumers (though uptake) and "producers" (through
        degradation enzyme production).
        Both maps are returned as (taxa, monomers) pandas DataFrames

        Return
        ------

            pandas.DataFrame
                taxon/monomer production map
            pandas.DataFrame
                taxon/monomer consumption map

        """
        tax_degenz = self.microbes.metabolites['Degradation_Enzymes'].map.astype(int)
        sub_degenz = self.degradation_enzymes.map.astype(int)
        sub_mon = self.substrate_degradation_op.map.astype(int)
        tax_sub = np.dot(tax_degenz, sub_degenz.T)
        tax_mon_prod = np.dot(tax_sub, sub_mon)
        tax_mon_prod = (tax_mon_prod > 0).astype(int)
        prod_df = pd.DataFrame(
                data=tax_mon_prod,
                index=self.microbes.names,
                columns=self.monomers.names,
                )
        #
        tax_upt = self.microbes.metabolites['Uptake_Transporters'].map.astype(int)
        mon_upt = self.monomer_uptake_op.map.astype(int)
        tax_mon_cons = np.dot(tax_upt, mon_upt.T)
        tax_mon_cons = (tax_mon_cons > 0).astype(int)
        cons_df = pd.DataFrame(
                data=tax_mon_cons,
                index=self.microbes.names,
                columns=self.monomers.names,
                )
        return prod_df, cons_df

    def save_biomass_fields_to_hdf5(self, filename):
        """
        Save all biomassfields that have been tagged with the 'save'
        tag to hdf5 file.

        Parameters
        ----------

            filename : str
                Path of the hdf5 file to create and write to

        """
        with h5py.File(filename, 'a') as hdf_file:
            for fname, bf in self.field_locator.biomassfields.items():
                tags = self.field_locator.biomassfields_tags[fname]
                if ('save' in tags):
                    # if (fname == 'substrates'):
                        # print('fname {}'.format(fname))
                        # print(tags)
                        # print(type(bf))
                        # print(np.sum(bf.mass))
                    bf._save_to_hdf_file(fname, hdf_file)

    def load_biomass_fields_from_hdf5(self, filename):
        """
        Load all biomassfields tagged with the 'save' tag from
        hdf5 file

        Parameters
        ----------
            filename : str
                Name of the hdf5 file to load data from
        """
        with h5py.File(filename, 'r') as hdf_file:
            for fname, bf in self.field_locator.biomassfields.items():
                tags = self.field_locator.biomassfields_tags[fname]
                if ('save' in tags):
                    bf._load_from_hdf_file(fname, hdf_file)


def ecosystem_from_parameters(pdict,
                              model_functions_module_name='model_functions',
                              model_functions_module_path=None,
                              ):
    """
    Build ecosystem state from parameters stored in a dictionnary.
    ( typically loaded from a saved json file)
    This restores system state except for biomassfields contents which
    must be loaded separately from hdf5 if one wants to restore full system
    state.

    Parameters
    ----------

        pdict : dict
            Dictionnary of ecosystem parameters
            (as obtained from :meth:`dementmutant.ecosystem.Ecosystem.get_metadata_dict`)

        model_functions_module_name : str
            Name of the python module from which to load model functions
            (it should be importable)

        model_functions_module_path : str or None
            Explicit path of the python file from which to import
            as the module containing model functions.

    Return
    ------

        :class:`dementmutant.ecosystem.Ecosystem`
            A  :class:`dementmutant.ecosystem.Ecosystem` instance with all parameters set from
            parameter dictionnary and all biomassfields set to zero.

    """
    eco = Ecosystem(
            grid_shape=pdict['grid_shape'],
            substrate_names=pdict['substrates']['names'],
            degradation_enzyme_names=pdict['degradation_enzymes']['enzyme_names'],
            monomer_names=pdict['monomers']['names'],
            uptake_transporter_names=pdict['uptake_transporters']['enzyme_names'],
            osmolyte_names=pdict['microbes']['metabolites']['Osmolytes']['names'],
            taxa_names=pdict['microbes']['taxa_names'],
            model_functions_module_name=model_functions_module_name,
            model_functions_module_path=model_functions_module_path,
                   )
    eco.set_parameters_from_metadata(pdict)

    return eco


def ecosystem_from_json_file(filename,
                             functions_module_name='model_functions',
                             functions_module_path=None,
                             ):
    """
    Build ecosystem state from parameters stored in a json file
    This restores system state except for biomassfields contents which
    must be loaded separately from hdf5 if one wants to restore full system
    state

    Parameters
    ----------
        filename : str
            Name of the json file from which to load parameters
        functions_module_name : str
            Name of the python module from which to load model functions
            (it should be importable)

        functions_module_path : str or None
            Explicit path of the python file from which to import
            as the module containing model functions.

    Return
    ------

        :class:`dementmutant.ecosystem.Ecosystem` instance with all parameters set from
        parameter dictionnary and biomassfields set to zero.

    """
    pdict = dict_from_json(filename)
    eco = ecosystem_from_parameters(pdict,
                                    functions_module_name,
                                    functions_module_path,
                                    )
    return eco


class BiomassFieldLocator:
    """

    This class keeps track of :class:`dementmutant.biomassfield.BiomassField` references.
    It allows to register named fields references, add arbitratry metadata tags
    to field references.
    This allows to maintain a flat database of allocated :class:`dementmutant.biomassfield.BiomassField`
    in the ecosystem object for

    - object serialization:  fields references are replaced by string denominators
      when saving parameters to json, and converted back to actual objects references
      when reloading

    - ecosystem saving : :class:`dementmutant.biomassfield.BiomassField` contents can be easily
      saved to disk without having to parse the whole object structure. Selection
      of fields to be saved is done using a single 'save' tag

    - diagnostic collection : the rudimentary :class:`DiagonosticCollector`
      uses a field locator and tags to produce :class:`dementmutant.biomassfield.BiomassField`
      based diagnostics


    """
    def __init__(self, parent):
        """
        Parameters
        ----------
            parent : :class:`dementmutant.ecosystem.Ecosystem`
                An instance of Ecosystem to attach to
        """
        #: :class:`dementmutant.ecosystem.Ecosystem`
        self.parent = parent
        #: dict of :class:`dementmutant.biomassfield.BiomassField`
        self.biomassfields = {}
        #: dict of list of str
        self.biomassfields_tags = {}

    def register(self, name, field, tags=None):
        """
        Append field reference to the field locator database

        Parameters
        ----------

            name : str
                Name under which the field is registered

            field : :class:`dementmutant.biomassfield.BiomassField`
                :class:`dementmutant.biomassfield.BiomassField` instance to register
            tags: list of str, optional
                List of string tags associated with the field

        """
        self.biomassfields[name] = field
        self.biomassfields_tags[name] = tags

    def get_fieldname(self, field):
        """
        Get name of field in the database if it is registered. Return None if it
        is not

        Parameters
        ----------
            field : :class:`dementmutant.biomassfield.BiomassField`
                Biomass field
        Return
        ------
            str
                Name of the field in the field locator database

        """
        for k, f in self.biomassfields.items():
            if (f == field):
                return k
        return None

    def get_field(self, fieldname):
        """
            Get field reference from its name. Return None if no field is registered
            under fieldname

        Parameters
        ----------
            fieldname : str
                Name of field to lookup

        Return
        ------
            :class:`dementmutant.biomassfield.BiomassField`
                requested biomassfield registered under fieldname
        """
        if (fieldname in self.biomassfields.keys()):
            return self.biomassfields[fieldname]
        else:
            print('Warning - unreferenced biomass field {}'.format(fieldname))
            return None

    def update(self):
        """
            Call parent (ecosystem) to populate database with reference.
            Naming of the various fields is delegated to the parent.
        """
        self.parent.update_field_locator()

    def field_to_str(self, data_in):
        """
        Recursively Convert field reference to their name in
        a nested dict, tuple structure.
        This is used for object serialization, as some classes parameters
        are user-defined references to fields, which must be reactivated on
        reload.

        Parameters
        ----------
            data_in : dict|tuple|:class:`dementmutant.biomassfield.BiomassField`
                data structure to convert, if iterable recursively calls
                itself on members.
        Return
        ------
            dict|tuple|str
                Initial object with :class:`dementmutant.biomassfield.BiomassField` replaced
                by their name with a special '__FIELD__' prefix
        """
        if (isinstance(data_in, dict)):
            data_out = {}
            for k, d in data_in.items():
                data_out[k] = self.field_to_str(d)
            return data_out
        elif isinstance(data_in, tuple):
            return tuple(self.field_to_str(i) for i in data_in)
        elif isinstance(data_in, BiomassField):
            data_name = self.get_fieldname(data_in)
            return '__FIELD__{}'.format(data_name)
        else:
            return data_in

    def _field_from_field_str(self, field_str):
        """
            Convert serialized field string to field name
        Parameters
        ----------
            field_str : str
                field description string as found in parameters dictionnaries
        Return
        ------
            str
                fieldname stripped from special prefix

        """
        assert(field_str[:9] == '__FIELD__')
        fieldname = field_str.strip('__FIELD__')
        return fieldname

    def add_tags_to_fields(self, new_tags, filter_func=None):
        """
        Append tags to a set of field references

        Parameters
        ----------
            new_tags : str|list of str
                tag or list of tags to add
            filter_func : function|None
                boolean filter function taking fieldname and current field tags as
                arguments. New tags are only added to when filter_func returns True.
        """
        if (isinstance(new_tags, str)):
            new_tags = [new_tags, ]
        for k in self.biomassfields_tags.keys():
            tl = self.biomassfields_tags[k]
            if (filter_func is not None):
                filt = filter_func(k, tl)
                if (not filt):
                    continue
            self.biomassfields_tags[k] = list(set(tl + new_tags))

    def get_metadata_dict(self):
        """
            Store metadata in a dictionnary for serialization to file.
            Only tags need to be saved, as biomassfields themselves are
            regenerated from the parent Ecosystem object

        Return
        ------
            dict
                dictionnary of attributes to save

        """
        a_list = ['biomassfields_tags', ]
        res = {}
        for a in a_list:
            res[a] = getattr(self, a)
        return res

    def set_parameters_from_metadata(self, pdict):
        """
            Reset tags from serialized dictionnary

        Parameters
        ----------
            pdict : dict
                dictionnary of saved fields tags
        """
        a_list = ['biomassfields_tags', ]
        for a in a_list:
            if ('biomassfields_tags' == a):
                for k in pdict[a].keys():
                    if (k not in self.biomassfields.keys()):
                        msg = 'WARNING loaded tags for unknown field {}'
                        print(msg.format(k))
            setattr(self, a, pdict[a])


class DiagnosticCollector:
    """

    A utility class to record and store diagnostic timelines for biomassfields

    **Diagnostic functions structure**

    The signature and return values of the provided function depend on the diagnostic type
    set by `diagtype`.

    Case of biomassfields diagnostic

    The diagnostic function takes a single :class:`dementmutant.biomassfield.BiomassField`
    and time index as arguments and returns a :class:`numpy.ndarray`
    (or array_like)
    (typically this will be the result of a reduction of the field
    data, either by space summation, mean, local value extraction...)

    Case of global diagnostics

    The diagnostic function takes a single :class:`dementmutant.ecosystem.Ecosystem`
    and time index as arguments.
    The return type should be a dictionnary whoses keys are the various
    quantities names and values are :class:`numpy.ndarray` (or array_like)
    For instance a diagnostic recording climatic data could return
    {'temperature': 21, 'psi': 1.5} or {'temperature': [21,], 'psi': [1.5,]}

    In both cases, if diag_func returns None for some time index, no data is recorded
    for this time index. Hence the user can take snapshots at arbitrary
    time indices by setting conditions on the time index into the diagnostic function

    :note:
        One should note that biomassfields diagnostics are a particular case.
        Their behaviour could perfectly be mimicked by a global diagnostic,
        which has access to the whole ecosystem data.
        In that case it would be up to the user writing the diagnostic function
        to properly reference the fields of interest.
        As single biomassfields diagnostics are expected to be massively used
        the 'fields' option allows to write simpler diagnostic functions.
        Looping over the various fields and selecting them is done automatically.

    """
    def __init__(self, ecosystem):
        """
        Parameters
        ----------

        ecosystem : :class:`dementmutant.ecosystem.Ecosytem`, optional
            Ecosystem object from which to record global diagnostics.

        """
        assert(isinstance(ecosystem, Ecosystem))
        #: :class:`dementmutant.ecosystem.Ecosystem` | None
        self.ecosystem = ecosystem
        #: :class:`dementmutant.ecosystem.BiomassFieldLocator`
        self.locator = ecosystem.get_field_locator()
        #: dict of dict
        self.timelines_desc = {}
        #: dict of dict
        self.timelines_records = {}
        #: dict of dict of dict of array_like
        self.timelines = {}

    def set_field_timeline_desc(self, name, diag_func, field_names='all'):
        """
        Define a timeline descriptor for biomassfields diagnostics

        Parameters
        ----------
            name : str
                A (human readable) name for the timeline

            diag_func : function
                The function used to extract diagnostics

            field_names : list of str | 'all'
                fields for which the diagnostic function must be applied and
                timelines generated

        """

        if (field_names == 'all'):
            fnames = list(self.locator.biomassfields.keys())
        elif (isinstance(field_names, list)):
            fnames = []
            for f in field_names:
                if (f not in self.locator.biomassfields.keys()):
                    print('Unknown field {} - omitting'.format(f))
                else:
                    fnames.append(f)

        else:
            print("field_names must be either 'all' or a list of fields")
            return
        self.timelines_desc[name] = {'fields': fnames,
                                     'func': diag_func,
                                     'source': 'fields'
                                     }

    def set_global_timeline_desc(self, name, diag_func):
        """
        Define a timeline descriptor for global diagnostics

        Parameters
        ----------
            name : str
                A (human readable) name for the timeline

            diag_func : function
                The function used to extract diagnostics

        """
        self.timelines_desc[name] = {'fields': [],
                                     'func': diag_func,
                                     'source': 'ecosystem'
                                     }

    def init_timelines(self):
        """
            Generate empty strucs for timelines from timelines descriptors
        """
        self.timelines_records = {}
        for k, d in self.timelines_desc.items():
            if (d['source'] == 'fields'):
                self.timelines_records[k] = {f: {} for f in d['fields']}
            else:
                self.timelines_records[k] = {}

    def record_single_timeline(self, tl_name, itime):
        """
            Record data for a single timeline
            Only useful if one want to perfom a record for a single
            timeline at some point of the time loop diffent from others
            recordings (for instance  an intermediary value that is not
            defined at the end of the time step)
            In the most common case recordings are taken simultaneously
            for all diagnostics at the end of the time step using
            :meth:`dementmutant.ecosystem.DiagosticCollect.record_timelines`

        Parameters
        ----------
            tl_name : str
                Name of the timeline diagnostic. It must have been
                previously defined using either `set_field_timeline_desc`
                or `set_global_timeline_desc`
            itime : int
                time index

        """

        if (tl_name in self.timelines_desc.keys()):
            k = tl_name
            d = self.timelines_desc[tl_name]
            if (d['source'] == 'fields'):
                for f in d['fields']:
                    diag = d['func'](self.locator.biomassfields[f], itime)
                    if (diag is not None):
                        self.timelines_records[k][f][itime] = diag
            elif (d['source'] == 'ecosystem'):
                diag_d = d['func'](self.ecosystem, itime)
                if (diag_d is not None):
                    for f, diag in diag_d.items():
                        if (f not in self.timelines_records[k].keys()):
                            self.timelines_records[k][f] = {}
                        self.timelines_records[k][f][itime] = diag
        else:
            print('Undefined timeline {}'.format(tl_name))

    def record_timelines(self, itime):
        """
        Apply diagnostic functions for all registered timelines and
        record data

        Parameters
        ----------

            itime : int
                time index

        """
        for k in self.timelines_desc.keys():
            self.record_single_timeline(k, itime)

    def set_timelines_from_records(self, tl_names=None):
        if (tl_names is None):
            tl_names = list(self.timelines_records.keys())
        elif isinstance(tl_names, str):
            tl_names = [tl_names, ]
        assert(all(isinstance(s, str) for s in tl_names))

        for tl_name in tl_names:
            res = {}
            if (tl_name not in self.timelines_records.keys()):
                print('Warning - requested unrecorded timeline {}'.format(tl_name))
                continue
            for f, fdiag in self.timelines_records[tl_name].items():
                times = np.array(list(fdiag.keys())).astype(_default_dtype)
                vals = np.row_stack(
                        [np.expand_dims(np.array(v, ndmin=1), axis=0) for v in fdiag.values()]
                        )
                res[f] = {'times': times, 'values': vals}
            self.timelines[tl_name] = res

    def get_timeline_arrays(self, tl_name, tag_filter_func=None):
        """
        For each field of a given timeline tl_name, concatenate
        individual recordings in a single :class:`numpy.ndarray`
        whose first axis is the time axis. Build a 1d :class:`numpy.ndarray`
        of time values.

        Parameters
        ----------
            tl_name : str
                name of the diagnostic timeline
            tag_filter_func : function, optional
                boolean selector function which take a list of tags as input
                Whenever the functions return false, the corresponding field
                is ignored.

        Return
        ------
            dict of 2-tuple (numpy.ndarray,numpy.ndarray)
                Dictionnary, keyed by field names, of tuples (times, diag)
                with times a 1d numpy.ndarray of time indexes at which diagnostics
                where computed, and diag a numpy.ndarray whose first axis matches
                times shape.


        """
        if (tl_name not in self.timelines.keys()):
            print('Setting tl from records {}'.format(tl_name))
            self.set_timelines_from_records([tl_name, ])

        res = {}
        for f, fdiag in self.timelines[tl_name].items():
            if (tag_filter_func is None):
                filt = True
            else:
                filt = tag_filter_func(self.locator.biomassfields_tags[f])
            if not(filt):
                continue
            res[f] = fdiag
        return res

    def append_timelines_to_hdf(self, tl_name, hdf_file):
        """
        Save concatenated timeline data to hdf5 file

        Parameters
        ----------
            tl_name : str
                Name of the timeline to save
            hdf_file : :class:`h5py.File`
                an open writable  :class:`h5py.File` instance
        """
        grp = hdf_file.require_group(tl_name)
        tl_dict = self.get_timeline_arrays(tl_name)
        for k, d in tl_dict.items():
            subgrp = grp.require_group(k)
            for dname, dat in d.items():
                if (dname in subgrp.keys()):
                    del(subgrp[dname])
                subgrp.create_dataset(name=dname, data=dat, compression='gzip')

    def dump_all_timelines_to_hdf5(self, hdf_filename):
        """
            Creates hdf5 file and save all timelines to it.

        Parameters
        ----------
            hdf_filename : str
                Path of the hdf5 file to create
        """
        self.set_timelines_from_records()
        hdf_file = h5py.File(hdf_filename, 'a')
        for tl_name in self.timelines.keys():
            self.append_timelines_to_hdf(tl_name, hdf_file)
        hdf_file.close()

    def load_from_dumpfile(self, hdf_filename):

        hdf_file = h5py.File(hdf_filename, 'r')
        for k, d in hdf_file.items():
            print('Loading timeline {}'.format(k))
            if (k not in self.timelines.keys()):
                self.timelines[k] = {}
            for k2, d2 in d.items():
                tmp_d = {}
                for k3, d3 in d2.items():
                    tmp_d[k3] = np.empty_like(d3)
                    tmp_d[k3][()] = d3[()]
                self.timelines[k][k2] = tmp_d
        hdf_file.close()

        return True


