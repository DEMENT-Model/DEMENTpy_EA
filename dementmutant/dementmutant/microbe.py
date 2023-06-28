#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Microbial entities and processes

Created on Fri Oct 22 14:40:01 2021

Copyright CNRS

@author: david.coulette@ens-lyon.fr

------------------------------------------------------------------------------

"""

import numpy as np
import pandas as pd
import inspect

from dementmutant.defaults import _default_dtype
from dementmutant.biomassfield import BiomassField
from dementmutant.utility import LHS


__microbes_are_quantized = False

def set_microbe_quantization_on():
    global __microbes_are_quantized
    __microbes_are_quantized = True

def set_microbe_quantization_off():
    global __microbes_are_quantized
    __microbes_are_quantized = False

def get_microbe_quantization():
    global __microbes_are_quantized
    return __microbes_are_quantized


class Metabolite:
    """
        Class for  storing per taxon metabolic parameters for various processes.
        It is meant to be used to store metabolic parameters such as production factors and costs, as well  as maintenance costs
        Two sets of factors are considered

            - constitutive : biomass transfers are relative to taxon biomass
            - inducible: biomass transfers are relative to taxon monomer accessible uptake

        For each set, to types of process targets are considered

            - Prod : biomass transferred to explicit metabolite production

            - Maintenance: biomass transferred to global C pools (CO2 ...)

        From a formal point of view the class simply stores coefficients
        for a 1st order linear ODE (vectorial) system of the form

        .. math::


           d_t T  = [A S]_{eff}

           d_t S  = - [A S]_{eff}

        with positivity contraint :math:`(S \ge 0 )`
        (namely the effective flux :math:`[A S]_{eff}` is the linear flux A S limited by reactant availability)
        with S the source of biomass and T the target
        The class stores a collection processes for which sources S are taxa
        biomasses, targets T are various biomass pools related to metabolic production
        and coefficients

        :todo: change name to MetabolicTrait(Set)  ? MetabolicGene(Set) ?
    """

    __taxa_source_atoms = 'C'  # biomass source channel
    __process_types = ['Induci', 'Consti']
    __target_types = ['Prod', 'Maint']
    __taxa_cons_atoms_full = ['C', 'N', 'P']
    __meta_prod_atoms_full = ['C', 'N', 'P']

    def __init__(self,
                 taxa_names, metabolite_names,
                 field_locator=None,
                 ):

        self.__ntaxa_atoms = len(self.__taxa_cons_atoms_full)
        self.__nprocess_types = len(self.__process_types)
        self.__ntarget_types = len(self.__target_types)
        self.__process_ids = {k: i for i, k in enumerate(self.__process_types)}
        self.__target_ids = {k: i for i, k in enumerate(self.__target_types)}
        #: list of str
        self.taxa_names = taxa_names
        #: list of str
        self.names = metabolite_names
        self.__metabolite_indexes = {k: i for i, k in enumerate(self.names)}
        #: int
        self.n_taxa = len(taxa_names)
        #: int
        self.n_metabolites = len(metabolite_names)
        #: :class:`numpy.ndarray` of bool, shape (n_taxa, n_metabolites)
        self.map = np.zeros((self.n_taxa, self.n_metabolites), dtype=bool)
        #: :class:`numpy.ndarray` of float, shape (n_taxa, n_metabolites)
        self.map_t = np.zeros((self.n_taxa, self.n_metabolites), dtype=_default_dtype)
        #: dict of :class:`numpy.ndarray` of float, shape (n_taxa, n_metabolites)
        self.weight_map  = {k1: np.ones((self.n_taxa,self.n_metabolites),
                                                dtype=_default_dtype)
                                   for k1 in self.__process_types
                                   }
        #: dict of :class:`numpy.ndarray` of float shape (n_taxa)
        self.ref_cost_v = {k1: np.zeros((self.n_taxa,), dtype=_default_dtype)
                           for k1 in self.__process_types
                           }
        #: dict of :class:`numpy.ndarray` of float shape (n_taxa,n_metabolites)
        self.ref_cost = {k1: np.zeros((self.n_taxa, self.n_metabolites), dtype=_default_dtype)
                         for k1 in self.__process_types
                         }
        #: dict of dict of :class:`numpy.ndarray` of float shape (n_atoms,)
        self.rel_cost = {k1: {k2: np.zeros((self.__ntaxa_atoms, ), dtype=_default_dtype)
                         for k2 in self.__target_types}
                         for k1 in self.__process_types
                         }
        #: dict of dict of bool
        self.active_pathway = {k1: {k2: False for k2 in self.__target_types}
                               for k1 in self.__process_types
                               }
        #: dict of dict of func
        self.environ_functions = {k1: {k2: None for k2 in self.__target_types}
                                  for k1 in self.__process_types
                                  }
        #: dict of dict of str
        self.environ_functions_names = {k1: {k2: "" for k2 in self.__target_types}
                                        for k1 in self.__process_types
                                        }
        #: dict of dict of :class:`dementmutant.biomassfield.BiomassField`
        self.targets = {k1: {k2: None for k2 in self.__target_types}
                        for k1 in self.__process_types}
        #: dict of dict of 2-tuple of int
        self.targets_grid_shapes = {k1: {k2: None for k2 in self.__target_types}
                                    for k1 in self.__process_types
                                    }

        self.external_targets = {k1: {k2: None for k2 in self.__target_types}
                                 for k1 in self.__process_types
                                 }
        #: dict of dict of :class:`dementmutant.biomassfield.BiomassField` or None
        self.target_priorities = {k1: {k2: 0 for k2 in self.__target_types}
                                  for k1 in self.__process_types
                                  }

        self.__functions_set = set()
        self.__field_locator = field_locator

    def get_metadata_dict(self):
        direct_attrs = ['taxa_names',
                        'names',
                        'active_pathway',
                        'external_targets',
                        'target_priorities',
                        'targets_grid_shapes',
                        'environ_functions_names'
                        ]
        res = {a: getattr(self, a) for a in direct_attrs}

        for a in ['map', ]:
            res[a] = getattr(self, a).tolist()
        for a in ['ref_cost_v', 'weight_map']:
            res[a] = {k: d.tolist() for k, d in getattr(self, a).items()}
        for a in ['rel_cost', ]:
            res[a] = {k1: {k2: d2.tolist() for k2, d2 in d1.items()}
                      for k1, d1 in getattr(self, a).items()
                      }
        return res

    def set_parameters_from_metadata(self, mdict, locator, functions_module):
        self.map = np.array(mdict['map'], dtype=bool)
        self.map_t = self.map.astype(_default_dtype)
        for k in self.ref_cost_v.keys():
            self.ref_cost_v[k][:] = np.array(mdict['ref_cost_v'][k])
        for k in self.weight_map.keys():
            self.weight_map[k][()] = np.array(mdict['weight_map'][k])
        self.update_ref_costs()
        for k1 in self.__process_types:
            for k2 in self.__target_types:
                if (not(mdict['active_pathway'][k1][k2])):
                    continue
                cv = np.array(mdict['rel_cost'][k1][k2])
                if (mdict['external_targets'][k1][k2] is None):
                    field_d = tuple(mdict['targets_grid_shapes'][k1][k2])
                else:
                    fname = mdict['external_targets'][k1][k2]
                    fieldname = locator._field_from_field_str(fname)
                    field_d = locator.get_field(fieldname)
                prio = mdict['target_priorities'][k1][k2]
                envf_name = mdict['environ_functions_names'][k1][k2]
                if (len(envf_name) > 0):
                    f = getattr(functions_module, envf_name)
                else:
                    f = None
                self.set_target(k1, k2, cv, field_d, prio, f)

    def get_functions_set(self):
        return self.__functions_set

    def get_target_types(self):
        """
            Return target type list
        """
        return self.__target_types

    def get_metabolite_index(self, metname):
        return self.__metabolite_indexes[metname]

    def update_ref_costs(self, processes=None):
        if (processes is None):
            plist = list(self.ref_cost.keys())
        else:
            if isinstance(processes, str):
                processes = [processes,]
            plist = [p for p in processes if p in self.ref_cost.keys()]
        for k in plist:
            self.ref_cost[k][:, :] = np.dot(np.diag(self.ref_cost_v[k]), self.map_t[:, :])
            self.ref_cost[k][:, :] = self.ref_cost[k][:, :] * self.weight_map[k]

    def set_map(self, map_in):
        assert(map_in.shape == self.map.shape)
        self.map[:,:] = map_in[:,:]
        self.map_t  = self.map.astype(self.map_t.dtype)

    def set_map_from_bounds(self, bounds, rng):
        """
        Draw taxon/metabolite assocication with only bounds

        Parameters
        ----------
            bounds: size 2 iterable of int
                (min,max) number of metabolie per taxon
            rng : numpy.random.generator
                A random generator used to draw the association map

        """
        # draw taxon metabolite association
        assert(bounds[0] >= 0)
        assert(bounds[1] <= self.n_metabolites)
        assert(bounds[0] <= bounds[1])
        #
        n_met = rng.choice(range(bounds[0], bounds[1]+1), size=self.n_taxa)
        #
        self.map[:, :] = False
        for itax in range(self.map.shape[0]):
            self.map[itax, :n_met[itax]] = True
            rng.shuffle(self.map[itax, :])
        self.map_t[:, :] = self.map.astype(self.map_t.dtype)

    def set_map_from_bounds_with_constraints(self, bounds, rng,
                                             forced_true=[],
                                             ):
        """
            Draw taxon/metabolite assocication with bounds and constraints

        Parameters
        ----------
            bounds: size 2 iterable of int
                (min,max) number of metabolie per taxon
            rng : numpy.random.generator
                A random generator used to draw the association map
            forced_true : list of str
                list of names of metabolites that should be associated to every taxon
        """
        # draw taxon metabolite association
        assert(bounds[0] >= 0)
        assert(bounds[1] <= self.n_metabolites)
        assert(bounds[0] <= bounds[1])
        #
        n_met = rng.choice(range(bounds[0], bounds[1]+1), size=self.n_taxa)
        #
        forced_filt = np.zeros((self.n_metabolites,), dtype=bool)

        for metname in forced_true:
            met_id = self.get_metabolite_index(metname)
            forced_filt[met_id] = True
        free_filt = np.logical_not(forced_filt)

        self.map[:, :] = False
        for itax in range(self.map.shape[0]):
            self.map[itax, forced_filt] = True
            tmp_f = np.zeros_like(self.map[itax, free_filt])
            tmp_f[:n_met[itax]] = True
            rng.shuffle(tmp_f)
            self.map[itax, free_filt] = tmp_f
        self.map_t[:, :] = self.map.astype(self.map_t.dtype)

    def set_ref_cost_v(self, process, val):
        self.ref_cost_v[process][:] = val
        self.update_ref_costs(process)

    def set_ref_costs_lhs(self, cost_bounds_dict, rng):
        """
            Draw taxon specific production efficiency

            Parameters
            ----------
                cost_bounds_dict : dict
                    Dictionnary of tuples of floats, keyed by process types
                    ('Consti','Induci').
        """
        assert(all([k in self.__process_types for k in cost_bounds_dict.keys()]))
        for k1 in self.__process_types:
            bounds = cost_bounds_dict[k1]
            if (bounds[1] > bounds[0]):
                self.ref_cost_v[k1][:] = LHS(self.n_taxa, bounds[0], bounds[1], 'uniform', rng=rng)
            elif (bounds[1] == bounds[0]):
                self.ref_cost_v[k1][:] = bounds[0] * np.ones((self.n_taxa,), dtype=_default_dtype)
            else:
                msg_str = 'Warning metabolic process {}  wrong bounds ()'
                print(msg_str.format(k1, bounds))
                self.ref_cost_v[k1][:] = np.zeros((self.n_taxa,), dtype=_default_dtype)
            self.update_ref_costs(k1)


    def set_weight_map(self, process, val):
        self.weight_map[process][()] = val
    def __get_2ddata_df(self, datname, typename=None):
        """
        Internal routine
        Get attribute array as 2d pandas DataFrame
        Parameters
        ---------
            datname : string
                data attribute
            typename : string
                If dat=ref_cost, process type (Consti, Induci)
        """
        if (datname in ['map', 'map_t']):
            dat = getattr(self, datname)
        elif (datname in ['ref_cost','weight_map']):
            dat = getattr(self,datname)[typename]
        else:
            print('Wrong parameters provided')
            return None
        res = pd.DataFrame(
                data=dat,
                index=self.taxa_names,
                columns=self.names,
                copy=True,
                )
        return res

    def get_map_bool_df(self):
        """
        Get tax/metabolite boolean association matrix

        Return
        ------
            pandas.Dataframe
                The taxon/metabolite association matrix as a pandas DataFrame
        """
        return self.__get_2ddata_df('map')

    def get_map_float_df(self):
        """
        Get tax/metabolite boolean association matrix as (0.0,1.0) float values

        Return
        ------
            pandas.Dataframe
                The taxon/metabolite association matrix as a real-valued (0,1) pandas DataFrame

        """
        return self.__get_2ddata_df('map_t')


    def get_weight_map_df(self, typename):
        """
        Get tax/metabolite boolean weight matrix

        Return
        ------
            pandas.Dataframe
                The taxon/metabolite association matrix as a real-valued pandas DataFrame

        """
        return self.__get_2ddata_df('weight_map', typename)

    def get_ref_cos_v_df(self, typename):
        """
        Get process reference cost per taxon multiplier

        Parameters
        ----------
            typename : string
                process type, either 'Induci' or 'Consti'
        """
        res = pd.DataFrame(
            data = self.ref_cost_v[typename],
            index = self.taxa_names,
            columns = ['Ref cost factor {}'.format(typename),],
            copy = True,
            )
        return res

    def get_ref_cost_df(self, typename):
        """
        Get process reference cost

        Parameters
        ----------
            typename : string
                process type, either 'Induci' or 'Consti'

        Return
        ------
            pandas.Dataframe
                The taxon/metabolite C-based cost matrix as a real-valued pandas DataFrame

        """
        self.update_ref_costs()
        return self.__get_2ddata_df('ref_cost', typename)



    def set_target(self, process, target, cost_v, field_d, priority=0, env_function=None):
        """
        Define the biomass pool target parameter for a given meatbolic process, ie
        the biomass field to which mass will be transfered

        Parameters
        ----------

            process : str
                process-type  either 'Consti' or 'Induci'
                (this defines the source for biomass transfer as either mass or delta_mass)

           target : str
                Target name (Prod, Maint) for production/maintenance

            cost_v : 1d array-like of floats
                cost/prod expressed as relative to C biomass

            field_d : :class:`biomassfield.BiomassField` or tuple of int
                Either
                    - an existing biomassfield (with suitable components)  which will be added
                      as an external target a internal biomass field with the same grid shape is
                      still created to keep acount of the transfer.
                    - a tuple (nx,ny) for grid shape, in that case a biomass field
                      object will be created for biomass reception.

        """
        dat = np.array(cost_v, dtype=_default_dtype)
        self.rel_cost[process][target][:] = dat[:]
        self.active_pathway[process][target] = True
        self.target_priorities[process][target] = priority
        if (isinstance(field_d, BiomassField)):
            self.external_targets[process][target] = field_d
            self.targets_grid_shapes[process][target] = field_d.grid_shape
            self.targets[process][target] = BiomassField(field_d.grid_shape,
                                                         self.names,
                                                         delta_buffers=['mass', ],
                                                         )
        elif(isinstance(field_d, tuple)):
            assert(len(field_d) == 2)
            self.targets_grid_shapes[process][target] = field_d
            self.targets[process][target] = BiomassField(field_d, self.names)
        if (env_function is not None):
            self.environ_functions[process][target] = env_function
            self.environ_functions_names[process][target] = env_function.__name__
            self.__functions_set = set(list(self.__functions_set) + [env_function, ])

        if (self.__field_locator is not None):
            self.__field_locator.update()

    def get_base_cost(self, process, target, env=None):
        """
        Computes and recover bases costs from parameters

        Parameters
        ----------
            process : str
                Process type ('Consti' or 'Induci')
            target : str
                Target type ('Prod' or 'Maint')
            env : dict (optional)
                environnental variable stored as a dict with keys ('temp','psi')

        Return
        ------
            numpy.ndarray

                A 3d numpy ndarray of shape (n_taxa, n_metabolites, na) containing the
                effective biomass cost factors
        """
        self.update_ref_costs()
        res = np.zeros((self.n_taxa, self.n_metabolites, self.__ntaxa_atoms),
                       dtype=_default_dtype
                       )
        res = self.ref_cost[process][:, :, np.newaxis] * self.rel_cost[process][target][:]

        if (self.environ_functions[process][target] is not None):
            if (env is not None):
                res = res * self.environ_functions[process][target](env)
        return res


class MicrobialMortality(BiomassField):
    """
    This class stores various parameters for microbial mortality processes
        - deterministic death by setting lower survival bounds on biomass (starvation thresholds)
        - stochastic death trough an environment-modulated probability function
    """

    def __init__(self, microbes, field_locator=None):
        """
        Parameters
        ----------
            microbes : :class:`microbe.Microbe`
                An instance of the :class:`microbe.Microbe` to which the mortality
                processes apply

            field_locator : :class:`ecosystem.BiomassFieldLocator`
                An instance of :class:`ecosystem.BiomassFieldLocator` object.
        """
        comp_names = ['MicrobialDeath', ]
        BiomassField.__init__(self, microbes.grid_shape,
                              comp_names,
                              delta_buffers=['mass'])
        #: :class:`dementmutant.biomassfield.BiomassField`
        self.microbes = microbes
        #: :class:`numpy.ndarray`
        self.starvation_thresholds = np.zeros((self.microbes.mass.shape[-1], ),
                                              dtype=self.microbes.mass.dtype)

        #: dict of :class:`numpy.ndarray` of floats
        self.death_proba_params = {}
        #: function | None
        self.death_proba_func = None
        #: str | None
        self.death_proba_func_name = ""
        #: :class:`dementmutant.biomassfield.BiomassField`
        self.dead_matter_pool = None
        #: str
        self.dead_matter_component_name = None
        #: int
        self.dead_matter_component_id = -1
        #: bool
        self.is_active = True

        self.__field_locator = field_locator

    def activate(self):
        """
        Switch on operator
        """
        self.is_active = True

    def deactivate(self):
        """
        Switch off operator
        """
        self.is_active = False

    def set_death_proba_func(self, death_proba_func):
        """
        Set the probability function for stochastic death

        Parameters
        ----------
            death_proba_func : function
                The death probability function. Its signature can have
                any number of real parameters. Per-taxon arrays for each
                parameter are stored internally as 1d-arrays and can be set
                by the user.
                If environmental parameters such as 'temp' (temperature) or 'psi' (water potential),
                are present in the function signature, their values will be updated at each time step
                by the corresponding environment values provided by the ecosystem.

        """
        sig = inspect.signature(death_proba_func)
        self.death_proba_params = {}
        n_taxa = self.microbes.n_taxa
        _mdtype = self.microbes.get_biomass_dtype()
        for pname in sig.parameters:
            self.death_proba_params[pname] = np.zeros((n_taxa,), dtype=_mdtype)
        self.death_proba_func = death_proba_func
        self.death_proba_func_name = death_proba_func.__name__

    def set_dead_matter_pool(self, field, component_name):
        """
        Set the biomass pool that is used as a target for dead microbial mass

        Parameters
        ----------
            field : biomassfield.BiomassField
                Reference to a biomass field object where dead mass should be transferred
            component_name : str
                name of the component of the target biomass field receiving dead biomass
        """
        self.dead_matter_pool = field
        self.dead_matter_component_name = component_name
        self.dead_matter_component_id = self.dead_matter_pool.get_component_index(component_name)
        if (self.__field_locator is not None):
            self.__field_locator.update()

    def set_mortality_parameter(self, pname, value, tfilter=None):
        """
        Set values for a parameter of the mortality probability function

        Parameters
        ----------
            pname : str
                Name of the parameter to set.
                Must exist in the mortality function signature
            value : float or 1d ndarray of floats
                Value(s) of the parameter to set.
                If a single float is provided, it is broadcasted to taxa.
                If an array is provided it must be a 1d array of shape (n_taxa,)
            tfilter : ndarray of bool (optionnal):
                Boolean filter of shape (n_taxa,) selecting taxa for which
                the parameter must be set.

        """
        if pname in self.death_proba_params.keys():
            if (tfilter is None):
                self.death_proba_params[pname][:] = value
            else:
                if (isinstance(value,  np.ndarray)):
                    self.death_proba_params[pname][tfilter] = value[tfilter]
                else:
                    self.death_proba_params[pname][tfilter] = value
        else:
            print('Unknown mortality parameter {}'.format(pname))

    def get_mortality_proba(self, env):
        """
        Computes mortality probabiliy

        Parameters
        ----------
            env : dict
                dictionnary containing environmental variables ('temp','psi')
        Return
        ------
            ndarray of floats
                Per-taxon death probability array

        """
        for k, d in env.items():
            if k in self.death_proba_params.keys():
                self.set_mortality_parameter(k, d)
        res = self.death_proba_func(**self.death_proba_params)
        return res

    def apply_transfer(self):
        """
        Add latest dead microbial mass to mass pool, and transfer it to
        external pool if necessary.
        """
        self.mass += self.delta_mass
        cid = self.dead_matter_component_id
        self.dead_matter_pool.mass[:, :, cid, :] += self.delta_mass[:, :, 0, :]
        self.delta_mass[()] = 0.0
        self.dead_matter_pool.update_biomass_fields()
        self.update_biomass_fields()

    def get_metadata_dict(self):
        res = {
                'death_proba_func_name': self.death_proba_func_name,
                'dead_matter_pool': self.dead_matter_pool,
                'dead_matter_component_name':self.dead_matter_component_name,
                'is_active': self.is_active,
                }
        res['death_proba_params'] = {k: d.tolist() for k, d in self.death_proba_params.items()}
        return res

    def set_parameters_from_metadata(self, pdict, locator, func_module):
        funame = pdict['death_proba_func_name']
        if (len(funame) > 0):
            func = getattr(func_module, funame)
            self.set_death_proba_func(func)

        fname = pdict['dead_matter_pool']
        if (len(fname) > 0):
            fieldname = locator._field_from_field_str(fname)
            field_d = locator.get_field(fieldname)
            self.dead_matter_pool = field_d

        self.dead_matter_component_name = pdict['dead_matter_component_name']
        self.dead_matter_component_id = self.dead_matter_pool.get_component_index(
            self.dead_matter_component_name)
        for k, d in pdict['death_proba_params'].items():
            self.death_proba_params[k][()] = np.array(d)
        self.is_active = pdict['is_active']


class StoechiometricBalanceRecycler(BiomassField):
    """
        This class stores parameters and provided routines for
        microbial biomass losses due to deviation from viable stoechiometry.


    """
    _taxa_types_id = {'undefined': 0, 'bacteria': 1, 'fungi': 2}

    def __init__(self, grid_shape, field_locator=None):
        """
        Parameters
        ----------

            grid_shape : 2-tuple of int
                Shape of space grid
            field_locator : :class:`ecosystem.BiomassFieldLocator`
                An Instance of a :class:`ecosystem.BiomassFieldLocator` object


            :todo:
                - check field_locator necessity/usage
        """
        comp_names = ['StoechiometricBalance', ]
        BiomassField.__init__(self, grid_shape, comp_names, delta_buffers=['mass'])
        #: dict of dict of 2-tuple of floats
        self.stoechiometry_quotas = {k: {k1: (0.0, 0.0) for k1 in self.get_atomnames('mass')}
                                     for k in self._taxa_types_id.keys()
                                     }
        #: dict of tuple (:class:`biomassfield.BiomassField`, str)
        self.transfer_pools = {k: None for k in self.get_atomnames('mass')}

        self.__field_locator = field_locator

        #: bool
        self.is_active = True

    def activate(self):
        """
        Switch on operator
        """
        self.is_active = True

    def deactivate(self):
        """
        Switch off operator
        """
        self.is_active = False

    def get_taxa_types(self):
        return list(self._taxa_types_id.keys())

    def set_stoechiometry_quotas(self, mic_type, quotas):
        """
        Set stoechiometry bounds for a given microbial type

        Parameters
        ----------

            mic_type : str
                A valid microbial type
            quotas : dict
                A dictionnary of 2-tuple of (optimal ratio,tolerance)  keyed by
                element name.
                Example : {'C':(0.8,0.1),'N':(0.1,0.0.05), 'P':(0.5, 0.1)}

        """
        if (mic_type not in self._taxa_types_id):
            print('Unknown microbe type {} ignoring'.format(mic_type))
            print('Available types : {}'.format(self.get_taxa_types()))
            return
        else:
            for atom, d in quotas.items():
                if (self.is_valid_atomname('mass', atom)):
                    if (isinstance(d, tuple)):
                        tup = d
                    else:
                        tup = (d, 0.0)
                    self.stoechiometry_quotas[mic_type][atom] = tup
                else:
                    print('Unkown atom name')

    def get_stoechiometry_quotas(self, type_filters, boundtype='min'):
        """
        Return stoechiometry quotas for each microbial type

        Parameters
        ----------

            type_filters : dict of 1d ndarray of bool
                Dictionnary of type filters, keyed by microbial type
                values are 1d boolean 1d-arrays of shape (n_taxa,)
                They are typically obtained from the relevant microbe instance.

            boundtype : str (default 'min')
                Type of bounds to compute from optimal and tolerance
                Must be either 'min', 'opt', 'max'

        Return
        ------
            numpy.ndarray
                A 2d numpy.ndarray of floats of shape (n_taxa,n_atoms)
                yielding mass ratio quota per taxon/per element

        """
        assert(boundtype in ['min', 'opt', 'max'])
        n_taxa = type_filters[list(type_filters.keys())[0]].shape[0]
        res = np.zeros((n_taxa, self.mass.shape[-1]), dtype=self.mass.dtype)
        for mic_type, filt in type_filters.items():
            for atom, d in self.stoechiometry_quotas[mic_type].items():
                iat = self.get_atom_index('mass', atom)
                v = max(0.0, {'min': d[0]-d[1], 'opt': d[0], 'max': d[0]+d[1]}[boundtype])
                res[filt, iat] = v
        return res

    def set_transfer_pool(self, atomname, field, compname):
        """
        Set up the biomass pool to which transfer excess biomass
        for a given element (atom)

        Parameters
        ----------

            atomname : str
                A valid atom name for the element being transfered
            field : :class:`biomassfielf.BiomassField`
                A :class:`biomassfielf.BiomassField` instance to which transfer
                element mass
            compname : str
                Name of the component of the target field towards which mass is transfered
        """
        fatoms = field.get_atomnames('mass')
        assert(atomname in fatoms)
        assert(atomname in self.transfer_pools.keys())
        assert(compname in field.names)
        self.transfer_pools[atomname] = (field, compname)
        if (self.__field_locator is not None):
            self.__field_locator.update()

    def apply_transfers(self):
        """
        Add computed mass transfer stored in delta_mass to mass pool
        and  add to external transfer pools.
        """
        f_l = []
        for k, fcomp in self.transfer_pools.items():
            if (fcomp is None):
                return
            field, compname = fcomp
            ia = self.get_atom_index('mass', k)
            ia_f = field.get_atom_index('mass', k)
            icomp = field.get_component_index(compname)
            field.mass[:, :, icomp, ia_f] += self.delta_mass[:, :, 0, ia]
            self.mass[:, :, 0, ia] += self.delta_mass[:, :, 0, ia]
            self.delta_mass[:, :, 0, ia] = 0.0
            f_l.append(field)
        f_l = list(set(f_l))
        for f in f_l:
            f.update_biomass_fields()

    def get_metadata_dict(self):
        return {
                'stoechiometry_quotas': self.stoechiometry_quotas,
                'transfer_pools': self.transfer_pools,
                'is_active': self.is_active,
                }

    def set_parameters_from_metadata(self, pdict, locator, func_module):

        for k1 in self.stoechiometry_quotas.keys():
            for k2 in self.stoechiometry_quotas[k1].keys():
                self.stoechiometry_quotas[k1][k2] = tuple(pdict['stoechiometry_quotas'][k1][k2])
        for k, tp in pdict['transfer_pools'].items():
            tpname, compname = tp
            if (len(tpname) > 0):
                fieldname = locator._field_from_field_str(tpname)
                field_d = locator.get_field(fieldname)
                self.transfer_pools[k] = (field_d, compname)
        self.is_active = pdict['is_active']


class MicrobialDispersal:
    """
    This class stores parameters and data buffers for
    microbial reproduction/dispersal

    In the DEMENT model, microbial entities are not described
    on an individual basis, but only through their biomass concentration
    per taxon per grid-cell.
    As a consequence reproduction is modeled using a discretization (quantification)
    procedure operating on biomass concentration.
    For each microbial species, a nominal concentration of a single element (typicall C) is
    used as a metric for the typical biomass content of a single individual.
    Whenever biomass concentration is above this threshold, biomass is split
    into one ore more quantas that can be dispersed.
    The exact mechanism for excess quantization and movement rules typically
    depends on microbial type.
    Two types of processes are implemented a of now

    - local_ratio : whenever the threshold is met, biomass for a given taxon
                    in a given grid cell is split, according to a predefined ratio
                    (typically 0.5/0.5), into two parts. One part stays in the cell,
                    while the other migrates. ( and optionnaly mutates if a mutation
                    operator is also applied)
                    This process is typically used as an approximation of bacterial mitosis.

    - non_local_sharing : excess biomass accross all cells on the grid for a given taxon
                          is pooled and redistributed across a live cells of the same
                          taxon.
                          This process is typically used as an approximation of fungal growth.

                          Note: as of now, spatial connexity of the live taxa network is not
                          taken into account, ie mass is redistributed accross the whole
                          grid.

    - quantized : base on number of individuals

    """

    _dispersal_types = ['local_ratio', 'non_local_sharing' , 'quantized', 'quantized_bdflux']

    def __init__(self, microbes):
        """
        Parameters
        ----------
            microbes : :class:`microbe.Microbe`
                An instance of the :class:`microbe.Microbe` on which
                the process is applied
        """
        #: :class:`dementmutant.microbe.Microbe`
        self.microbes = microbes
        taxa_types = self.microbes.get_taxa_types()
        self._taxa_types = taxa_types
        self._space_axes_names = list(self.microbes.get_space_axes_names())
        self.n_space_dims = len(self._space_axes_names)

        self.saturation_ref_atom = {k: None for k in taxa_types}
        #: :class:`numpy.ndarray` of float
        self.saturation_thresholds = np.empty((self.microbes.n_taxa,), dtype=_default_dtype)
        self.saturation_thresholds_dict = {k: None for k in taxa_types}
        self.saturation_thresholds[:] = np.infty
        self.biomass_ratios = {k: None for k in taxa_types}
        self.types = {k: None for k in taxa_types}
        self.forced_probas = {k0: {k: None for k in self._space_axes_names}
                              for k0 in taxa_types
                              }
        #: dict of dict of 2-tuple of int
        self.dranges = {k0: {k: (0, 0) for k in self._space_axes_names}
                        for k0 in taxa_types
                        }
        #
        shift_shape = (self.n_space_dims,
                       self.microbes.grid_shape[0],
                       self.microbes.grid_shape[1],
                       self.microbes.n_taxa)
        #: :class:`numpy.ndarray` of int
        self.shifts = np.zeros(shift_shape, dtype=int)

        disp_shape = (self.microbes.grid_shape[0],
                      self.microbes.grid_shape[1],
                      self.microbes.n_taxa,
                      )
        #: :class:`numpy.ndarray` of bool
        self.dispersion_filter = np.zeros(disp_shape, dtype=bool)

        #: bool
        self.is_active = True

        #: dict
        self.last_run_stats = {}
    def activate(self):
        """
        Switch on operator
        """
        self.is_active = True

    def deactivate(self):
        """
        Switch off operator
        """
        self.is_active = False

    def get_dispersal_types(self):
        """
        Return list of implemented dispersal mechanisms.

        Return
        ------
            list of string
                List of dispersal processes types
        """
        return self._dispersal_types

    def set_parameters(self,
                       mic_type,
                       disp_type,
                       disp_ratio,
                       disp_dist_ranges,
                       disp_forced_probas={},
                       ):
        """
        Define parameters for microbial colonization/dispersal
        Parameter sets are defined per microbial type.

        Parameters
        ----------
            mic_type : string
                A valid microbial type for the parent microbe object

           disp_type : string
               A valid dispersion process type, either of
                 - 'local_ratio' : a fraction of biomass is taken out of saturated cells
                 - 'non_local_sharing': biomass is pooled across grid before excess sharing

           disp_ratio : float
               Fraction of biomass removed from saturated cells.
               Used by the local_ratio process.
               Must be in :math:`[0,1]`

            disp_dist_ranges: dict
               Dictionnary of tuples, keyed by axis names ('x','y')
               Each tuple specify the minimal and maximal shifts along the corresponding axis

           disp_forced_probas : dict of dict of floats
               impose probability values for some values of shift distance
               highest-level keys : axe name ('x','y')
               second-level :  d:p pairs with d integer shift and p probability

        """
        if (mic_type not in self._taxa_types):
            print('Unknown microbe type {}'.format(mic_type))
            print('Valid types {}'.format(self._taxa_types))
            return
        if (disp_type not in self.get_dispersal_types()):
            print('Unknown dispersal type {}'.format(disp_type))
            print('Valid types {}'.format(self._dispersal_types))
            return
        if (disp_ratio < 0.0) or (disp_ratio > 1.0):
            print('Biomass dispersal ratio must be in [0,1]')
            return

        self.types[mic_type] = disp_type

        self.biomass_ratios[mic_type] = disp_ratio
        for k, d in disp_dist_ranges.items():
            self.dranges[mic_type][k] = d
        for k, d in disp_forced_probas.items():
            self.forced_probas[mic_type][k] = d

    def set_saturation_threshold(self, mic_type, atomname, threshold):
        """
        Define per taxon/per cell nominal individual size threshold above which dispersal occurs.
        for a given microbial type.

        Parameters
        ----------
            mic_type : string
                A valid microbial type

            atomname : string
                The element (atom) concentration field used as reference for size

            threshold  :float
                biomass concentration threshold value

        """
        if (not self.microbes.is_valid_atomname('mass', atomname)):
            print('Unknown atom name ')
            return
        if (mic_type in self._taxa_types):
            filt = self.microbes.type_filters[mic_type]
            self.saturation_ref_atom[mic_type] = atomname
            self.saturation_thresholds[filt] = threshold
            self.saturation_thresholds_dict[mic_type] = (atomname, threshold)
        else:
            print('Unknown microbe type {}'.format(mic_type))

    def get_dispersal_prob_distrib(self, mic_type):
        """
        Recover the probabilities matrix of dispersal shift along each direction
        for a given microbial type

        Parameters
        ----------
            mic_type  : string
                Valid microbial type

        Return
        ------
            dict of dict
                A direction-keyed dictionnary of dictionnaries containing
                shift values along spatial axes and their probabilities.

        """
        res = {}
        for kdim in self._space_axes_names:
            drange = self.dranges[mic_type][kdim]
            vals = np.array(list(range(drange[0], drange[1]+1)), dtype=int)
            probs = np.zeros(vals.shape, dtype=_default_dtype)
            fprob = np.zeros(vals.shape, dtype=bool)
            sump = 1.0
            forced_prob = self.forced_probas[mic_type][kdim]
            if forced_prob is not None:
                for d, p in forced_prob.items():
                    w = np.where(vals == int(d))
                    if (len(w[0]) > 0):
                        probs[w] = p
                        fprob[w] = True
                        sump -= p
                    else:
                        print('Forced proba for {} which is not in range'.format(d))
            assert(sump >= 0.0)
            free_probs = np.logical_not(fprob)
            nfree = np.sum(free_probs.astype(int))
            pfree = sump / nfree
            probs[np.nonzero(free_probs)] = pfree
            res[kdim] = {'vals': vals, 'probas': probs}
        return res

    def get_metadata_dict(self):
        att_l = ['saturation_ref_atom',
                 'biomass_ratios',
                 'types',
                 'forced_probas',
                 'dranges',
                 'saturation_thresholds_dict',
                 'is_active',
                 ]
        res = {}
        for a in att_l:
            res[a] = getattr(self, a)
        return res

    def set_parameters_from_metadata(self, pdict, locator, func_module):
        for k, d in pdict.items():
            setattr(self, k, d)

        tmp_d = {k:d for k,d in self.saturation_thresholds_dict.items()}
        for k,d in tmp_d.items():
            atomname,threshold = d
            self.set_saturation_threshold(k, atomname, threshold)


class MicrobialMutation:
    """
    This class stores parameters and data buffers for mutation
    processes.
    It is used in conjunction with the microbial :class:`microbe.MicrobialDispersal`
    class to described mutations as a stochastic movement along the taxon axis.

    Note that the genetic state space is finite and equal to the number of taxa.
    (some of which might be non-existent at some point in time)

    """
    def __init__(self, microbes):
        """
        Setup microbial mutation process

        Parameters
        ----------
            microbes : :class:`microbe.Microbe`
                An instance of the :class:`microbe.Microbe` class, on
                which the process applies

        """
        #: :class:`dementmutant.microbe.Microbe`
        self.microbes = microbes

        self._taxa_types = self.microbes.get_taxa_types()

        self._space_axes_names = list(self.microbes.get_space_axes_names())

        #: int
        self.n_space_dims = len(self._space_axes_names)

        #: bool
        self.is_active = False

        #: string in ['one','two']
        self.mutation_n_daughters = 'one'

        #: :class:`numpy.ndarray` of float shape (n_taxa,n_taxa)
        self.mutation_probas = np.identity(self.microbes.n_taxa, dtype=_default_dtype)

        shift_shape = (
                       self.microbes.grid_shape[0],
                       self.microbes.grid_shape[1],
                       self.microbes.n_taxa
                       )
        #: :class:`numpy.ndarray` of float shape (n_x, n_y, n_taxa)
        self.shifts = np.zeros(shift_shape, dtype=int)

        #: function
        self.mutation_prob_func = None

        #: dict
        self.mutation_prob_func_params = None

        #: dict
        self.last_run_stats = {}

    def activate(self):
        """
        Allows for mutations to occur
        """
        self.is_active = True

    def deactivate(self):
        """
        Prevents mutations from occuring
        """
        self.is_active = False

    def set_mutation_proba_func(self, func):
        """
        Define the mutation probability function.

        Parameters
        ----------
            func : function
                Mutation probability function with signature (microbes, proba, params)
                where microbes is :class:`microbe.Microbe` object
                proba is a (n_taxa,n_taxa) ndarray of floats, apd param is
                an optional dictionnary containing tuning parameters.
                The function should populate proba so that proba[i,j]
                is the probability for taxon i to mutate to taxon j
                The row [i,:] is the discrete probability distribution for the
                mutations of taxon i.
                As a consequence
                - all entries should be in [0,1]
                - the sum of each row should be one.

                NB : the user provided function **must** return proba
                ( the buffer is modified in-place)
        """
        sig = inspect.signature(func)

        valid_sigs = [{'microbes','proba'},{'microbes','proba','params'}]
        # if (set(list(sig.parameters.keys())) != set(['microbes', 'proba'])):
        if (set(list(sig.parameters.keys())) not in valid_sigs):
            msg_str = 'Wrong mutation probability signature \n Args should be {} or {}'
            print(msg_str.format(valid_sigs[0],valid_sigs[1]))
            assert(False)
        else:
            self.mutation_prob_func = func

    def check_proba_matrix(self):
        """
            Checks wether transition probability matrix is well build
        """
        chksums = np.sum(self.mutation_probas, axis=1)
        chk = np.all(chksums == 1.0)
        chk = chk and np.all(self.mutation_probas >= 0.0)
        chk = chk and np.all(self.mutation_probas <= 1.0)
        assert(chk)

    def compute_mutation_probas(self):
        """
        Applies user provided function to compute mutation probabilities
        """
        if (self.mutation_prob_func is not None):
            args = {'microbes': self.microbes, 'proba': self.mutation_probas}
            if (self.mutation_prob_func_params is not None):
                args['params'] = self.mutation_prob_func_params
            self.mutation_probas = self.mutation_prob_func(**args)

    def compute_taxa_shifts(self, rng):
        """
        Draw mutation transitions using transition probability matrix

        Parameters
        ----------

            rng : numpy.random generator
                The numpy.random generator to use for drawing taxa shifts

        """
        ntax = self.microbes.n_taxa
        pmat = self.mutation_probas
        tshifts = self.shifts
        dmass = self.microbes.delta_mass
        draw_shape = tshifts.shape[:2]
        for itax in range(self.microbes.n_taxa):
            tshifts[:, :, itax] = rng.choice(ntax,
                                             size=draw_shape,
                                             p=pmat[itax, :],
                                             replace=True)
            tshifts[:, :, itax][dmass[:, :, itax, 0] == 0] = itax

    def get_metadata_dict(self):
        if (self.mutation_prob_func is not None):
            fname = self.mutation_prob_func.__name__
        else:
            fname = ""
        res = {
               "is_active": self.is_active,
               'mutation_n_daughters': self.mutation_n_daughters,
               "mutation_prob_func_name": fname,
               "mutation_prob_func_params": self.mutation_prob_func_params
               }


        return res

    def set_parameters_from_metadata(self, pdict, locator, func_module):
        self.is_active = pdict["is_active"]
        if ('mutation_n_daughters' in pdict.keys()):
            self.mutation_n_daughters = pdict['mutation_n_daughters']
        fname = pdict['mutation_prob_func_name']
        if (len(fname) > 0):
            self.mutation_prob_func = getattr(func_module, fname)
        else:
            self.mutation_prob_func = None
        if ('mutation_prob_func_params' in pdict.keys()):
            self.mutation_prob_func_params = pdict['mutation_prob_func_params']


class Microbe(BiomassField):
    """
    Microbial (bacteria, fungi, whatever...) entities class

    This class stores microbial biomass as well as various parameters
    directly related to microbial processes.

    - metabolic processes (stored in a dictionnary)
    - mortality
    - stoechiometry balance
    - dispersal
    - mutations

    """
    _taxa_types_id = {'undefined': 0, 'bacteria': 1, 'fungi': 2}

    def __init__(self,
                 grid_shape,
                 taxa_names,
                 delta_buffers=[],
                 ):
        """
        Parameters
        ----------

            grid_shape : 2-tuple of int
                Shape of space grid
            taxa_names : list of str
                List of of taxa names. Its length defines the number of taxa
            delta_buffers: list of str
                list of biomassfield attributes for which a secondary buffer
                should be allocated
        """

        mic_quant = get_microbe_quantization()
        if (mic_quant and 'quanta' not in delta_buffers):
            delta_buffers += ['quanta',]
        BiomassField.__init__(self, grid_shape, taxa_names, delta_buffers=delta_buffers, is_quantized=mic_quant)
        #: int
        self.n_taxa = self.n_components
        #: dict of int
        self.tax_ids = {k: i for i, k in enumerate(self.names)}
        #: :class:`numpy.ndarray` of int shape (n_taxa,)
        self.taxa_types = np.zeros((self.n_taxa,), dtype=np.int_)  # undefined taxa by default
        #: int
        self.n_taxa_types = len(self._taxa_types_id)
        self._set_types_filters()

        #: dict of :class:`dementmutant.microbe.Metabolite`
        self.metabolites = {}

        # mortality related parameters

        #: :class:`dementmutant.microbe.MicrobialMortality`
        self.mortality_op = MicrobialMortality(self)

        #: :class:`dementmutant.microbe.StoechimetricBalanceRecycler`
        self.stoechio_balance_recycler = StoechiometricBalanceRecycler(self.grid_shape)

        # dispersal/colonization

        #: :class:`dementmutant.microbe.MicrobialDispersal`
        self.dispersal_op = MicrobialDispersal(self)
        # mutation

        #: :class:`dementmutant.microbe.MicrobialMutation`

        self.mutation_op = MicrobialMutation(self)

    def _set_types_filters(self):
        """
        Build a dictionnary of boolean arrays
        """
        self.type_filters = {k: (self.taxa_types == d) for k, d in self._taxa_types_id.items()}

        self.type_filters_df = pd.DataFrame.from_dict(self.type_filters)
        self.type_filters_df.index = self.names

    def get_taxa_types(self):
        """
        Return a list of valid microbial types

        Return
        ------
            list of str
                List of allowed taxa types
        """
        return list(self._taxa_types_id.keys())

    def set_types(self, type_name, taxa_names=[]):
        """
        Assign a microbial type to taxa

        Parameters
        ----------
            type_name : string
                A valid microbial type
            taxa_names (optional) : list of str
                List of taxa name to which to assign the microbial type.
                If not provided, the type will be assigned to all taxa.

        """
        if (type_name in self._taxa_types_id.keys()):
            typ_id = self._taxa_types_id[type_name]
        else:
            print('Unkown microbial type, choose among {}'.format(self._taxa_types_id.keys()))
            return
        if (taxa_names == []):
            self.taxa_types[:] = typ_id
        else:
            assert(all([t in self.tax_ids.keys() for t in taxa_names]))
            for tname in taxa_names:
                t_id = self.tax_ids[tname]
                self.taxa_types[t_id] = typ_id
        self._set_types_filters()

    def get_metabolites_names(self):
        """
        Return a list of currently defined metabolic entities

        Return
        ------
            list of str
            List of currently defined metabolic entities
        """
        return list(self.metabolites.keys())

    def get_metabolite(self, metasetname):
        """
        Return a metabolic entity object, defining a set of metabolic processes
        with common properties

        Parameters
        ----------
            metasetname : str
                name of the metabolic entity.

        Return
        ------
            :class:`microbe.Metabolite`
                An instance of the :class:`microbe.Metabolite` class

        """
        if (metasetname in self.metabolites.keys()):
            return self.metabolites[metasetname]
        else:
            print('Unknown metabolite entity')
            print('Defined entities are {}'.format(self.get_metabolites_names()))
            return None

    def add_metabolite(self, metasetname, metanames):
        """
        Append new metabolic entity to the pool of metabolic processes
        Parameters
        ----------
            metasetname : string
                name of the set of metabolic entities
            metanames : list of string
                list of names of metabolites in the set.

        """
        self.metabolites[metasetname] = Metabolite(self.names, metanames)

    def get_metabolic_processes_operands(self, process, env=None):
        """
        Get environment dependent data necessary for the computation
        of all the metabolic processes of a given type (Consititutive or Inducible)
        collated as lists

        Parameters
        ----------
            process : string
                process type ('Consti' or 'Induci')
            env (optionnal): dict
                environnment variables in a dictionnary with 'temp','psi' keys for temperature and water potential

        Processes for a given type are sorted by priority and assigned a rank
        ( 0 served first, 1 next, etc). Available biomass is served rank by rank
        Processes of rank 0 are served first, proportionnaly to what they would get
        with unlimited resources ( stoechiometric total reaction)
        Processes of rank 1 are the served using remaing biomass.
        And so on...

        Return
        ------
            target costs : list
                list of ndarrays of shape (n_taxa, n_metabolites, n_atoms) of
                cost for each process
            taxa costs :ndarray of floats
                (n_ranks, n_taxa, n_atoms) ndarray of  summed costs per priority rank, per taxa
            target_ranks : list of int
                list of priority rank of processes
        """
        prank_d = self.get_metabolic_processes_prio_rank_dict(process)
        n_ranks = len(prank_d)
        taxa_costs_shape = (n_ranks, self.n_taxa, self._BiomassField__biomass_n_atoms)
        taxa_costs = np.zeros(taxa_costs_shape, dtype=_default_dtype)
        target_costs = []
        target_ranks = []
        for metname, met in self.metabolites.items():
            for targ in(met.get_target_types()):
                if (met.active_pathway[process][targ]):
                    tmp_v = met.get_base_cost(process, targ, env)
                    target_costs.append(tmp_v)
                    priority = met.target_priorities[process][targ]
                    i_rank = prank_d[priority]
                    target_ranks.append(i_rank)
                    taxa_costs[i_rank, :] += np.sum(tmp_v, axis=1)
        return target_costs, taxa_costs, target_ranks

    def get_metabolic_processes_targets_old(self, process):
        """
        Return a list of reference of mass fields for a given process type

        Parameters
        ----------
            process : string
                process type ('Consti', 'Induci')

        Return
        ------
            list
                A list of ndarrays (mass fields)

        :todo:
            - DEPRECATE
        """
        res = []
        for metname, met in self.metabolites.items():
            for targ in(met.get_target_types()):
                if (met.active_pathway[process][targ]):
                    res.append(met.targets[process][targ].mass)
        return res

    def get_metabolic_processes_targets(self, process):
        """
        Return a list of reference of mass fields for a given process type

        Parameters
        ----------
            process : string
                process type ('Consti', 'Induci')

        Return
        ------
            list
                A list of ndarrays (mass fields)
        """
        res = []
        for metname, met in self.metabolites.items():
            for targ in(met.get_target_types()):
                if (met.active_pathway[process][targ]):
                    if (met.external_targets[process][targ] is None):
                        res.append(met.targets[process][targ].mass)
                    else:
                        res.append(met.targets[process][targ].delta_mass)
        return res

    def clear_metabolic_duplicate_targets_buffers(self, process):
        """
        Set metabolic targets delta buffers to 0
        """
        for metname, met in self.metabolites.items():
            for targ in(met.get_target_types()):
                if (met.active_pathway[process][targ]):
                    if (met.external_targets[process][targ] is not None):
                        met.targets[process][targ].delta_mass[()] = 0

    def transfer_metabolic_duplicate_targets(self, process):
        """
        Transfer mass increment to external targets
        """
        # TODO ? numba kernel implementation (unlikely to be useful except
        # for very large data + parallelisation)
        for metname, met in self.metabolites.items():
            for targ in(met.get_target_types()):
                if (met.active_pathway[process][targ]):
                    if (met.external_targets[process][targ] is not None):
                        delta = met.targets[process][targ].delta_mass
                        met.targets[process][targ].mass += delta
                        met.external_targets[process][targ].mass += delta

    def get_metabolic_processes_prio_rank_dict(self, process):
        """
        Build prioririty/rank association for a given process type

        Parameters
        ----------
            process : string
                process type ('Consti','Induci')

        Among a process class (Inducible or Constitutive) processes
        are assigned priorities by the used in the form of integer
        values ( the highest integer , the highest priority)
        This routine collate priorities and assign ranks so that

            - higher priorites have the lowest rank (served first)
            - equal priorities have the same rank

        Return
        ------
            dict
                dictionnary of ranks keyed by process priority
        """
        res = []
        prio = []
        for metname, met in self.metabolites.items():
            for targ in(met.get_target_types()):
                if (met.active_pathway[process][targ]):
                    prio.append(met.target_priorities[process][targ])
        prio = reversed(sorted(list(set(prio))))
        res = {p: r for r, p in enumerate(prio)}
        return res

    def sync_targets(self, process):
        """
        Trigger computation of total mass and biomass ratios for targets
        of a given process type

        Parameters
        ----------
            process : string
                process type (either 'Consti' or 'Induci')
        """
        for metname, met in self.metabolites.items():
            for targ in(met.get_target_types()):
                if (met.active_pathway[process][targ]):
                    met.targets[process][targ].update_biomass_fields()

    def sync_external_targets(self, process):
        """
        Update total mass and mass ratios of external target
        for a given process type

        Parameters
        ----------
            process : string
                process type (either 'Consti' or 'Induci')

        """
        for metname, met in self.metabolites.items():
            for targ in (met.get_target_types()):
                if (met.active_pathway[process][targ]):
                    tf = met.external_targets[process][targ]
                    if (tf is not None):
                        tf.update_biomass_fields()

    def is_external_metabolic_target(self, metname, process, target):
        """
            Test whether the target for a given process type and a given metabolic
            entities set is external.

        Parameters
        ----------
            metname : string
                A valid metabolic process name
            process : string
                A valid process type , either 'Consti' or 'Induci'
            target: string
                A valid target name ('Prod', 'Maint')

        Return
        ------
            bool
                True if the target is external (ie mass is transfered)
        """
        met = self.metabolites[metname]
        res = met.external_targets[process][target] is not None
        return res

    def get_targets_biomass_space_totals_dict(self, process):
        """
        Collects spaced-sums of active metabolic processes targets
        for a given process type

        Parameters
        ----------

        process : string
            process type ('Consti','Induci')

        Return
        ------
            dict of dict of ndarrays
                dictionnary of dictionnaries of (n_componenets, n_atoms) ndarrays
                highest level keys are process names
                second level keys are target types ('Prod', 'Maint')
        """
        res = {}
        for metname, met in self.metabolites.items():
            tmp_res = {}
            for targ in(met.get_target_types()):
                if (met.active_pathway[process][targ]):
                    t = met.targets[process][targ]
                    tmp_res[targ] = np.sum(t.get_space_total(), axis=0)
            if (len(tmp_res) > 0):
                res[metname] = tmp_res
        return res

    def set_stoechiometry_quotas(self, mic_type, quotas):
        """
        Defines stoechiometry quotas for microbes of a given type

        Parameters
        ----------
            mic_type : string
                A valid microbial type
            quotas : dict
                A dictionnary of stoechiometric ratios, keyed by atom name
        """
        self.stoechio_balance_recycler.set_stoechiometry_quotas(mic_type, quotas)

    def get_taxa_stoechiometry_quotas(self, boundtype='min'):
        """
        Get per taxa stoechiometry quotas

        Parameters
        ----------
            boundtype : string
                Type of reference required : either 'opt', 'min', 'max'

        Return
        ------
            numpy.ndarray
                (n_taxa,n_atoms) 2d-ndarray of mass ratios

        """
        res = self.stoechio_balance_recycler.get_stoechiometry_quotas(
                                                                      self.type_filters,
                                                                      boundtype,
                                                                      )
        return res

    def get_random_taxon_selection(self, prob, rng):
        shap = (self.grid_shape[0], self.grid_shape[1], self.n_components)
        res = rng.choice([0, 1], size=shap, replace=True, p=[1-prob, prob])
        return res

    def set_random_masses_from_quotas(self, bounds, rng, boundtype='opt', ref_scale='T'):
        """
        Set microbiall biomass fields with bounded random total mass
        and using stoechiometry bounds

        Parameters
        ----------
            bounds : tuple of floats
                bounds for total biomass per cell
            rng  :numpy.random generator
                random generator to use
            boundtype (optional) : string
                select stoechiometry reference
                ( default is 'opt', optimal one, ie representative of microbial typical
                mass ratios)

        """
        sbr = self.stoechio_balance_recycler
        quotas = sbr.get_stoechiometry_quotas(self.type_filters, boundtype)
        self.set_random_masses_from_bounds(bounds, rng, stoechio=quotas,ref_scale=ref_scale)

    def get_metadata_dict(self):
        res = {
                'grid_shape': self.grid_shape,
                'taxa_names': self.names,
                'taxa_types': self.taxa_types.tolist()
               }
        if hasattr(self, 'is_quantized'):
            res['is_quantized'] = self.is_quantized
            res['quantization_norm'] = self.quantization_norm.tolist()
            res['quantization_atom_id'] = self.quantization_atom_id
        meta_dict = {}
        for metname, met in self.metabolites.items():
            meta_dict[metname] = met.get_metadata_dict()
        res['metabolites'] = meta_dict
        op_list = ['mortality_op', 'stoechio_balance_recycler', 'dispersal_op', 'mutation_op']
        for a in op_list:
            o = getattr(self, a)
            res[a] = o.get_metadata_dict()
        return res

    def set_parameters_from_metadata(self, pdict, locator, functions_module):
        self.taxa_types[:] = np.array(pdict['taxa_types'])
        self._set_types_filters()
        meta_dict = pdict['metabolites']
        if ("is_quantized" in pdict.keys()):
            self.is_quantized = pdict["is_quantized"]
            self.quantization_atom_id = int(pdict["quantization_atom_id"])
            self.quantization_norm = np.array(pdict["quantization_norm"])
        for metname, met in self.metabolites.items():
            met.set_parameters_from_metadata(meta_dict[metname],
                                             locator,
                                             functions_module
                                             )
        locator.update()
        op_list = ['mortality_op', 'stoechio_balance_recycler', 'dispersal_op', 'mutation_op']
        for a in op_list:
            o = getattr(self, a)
            o.set_parameters_from_metadata(pdict[a], locator, functions_module)

        locator.update()

    def get_functions_set(self):
        res = set()
        for metname, met in self.metabolites.items():
            tmp_set = met.get_functions_set()
            res = set(list(res) + list(tmp_set))
        tmp_set = self.mortality_op.get_functions_set()
        res = set(list(res) + list(tmp_set))
        return res
