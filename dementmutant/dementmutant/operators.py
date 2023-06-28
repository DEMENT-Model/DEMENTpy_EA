#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Operators for dynamical processes - stable interface

Created on Thu Dec 16 14:30:12 2021

Copyright CNRS

@author: david.coulette@ens-lyon.fr

"""

import numpy as np

import dementmutant.operators_numba as operators_numba
import dementmutant.operators_numpy as operators_numpy
from dementmutant.defaults import _default_dtype

# routines from higher-level modules ( ecosystem, microbes etc)
# call computing kernels from this module
# in this module no implementations are provided, but links to
# implementations in others modules are sets
# the main purpose is to allow easily switching computation kernels
# without touching higher-level code
# for now the association is done statically (by the assignations below)
# a switch to a function factory design is under evaluation, to allow
# for easy dynamical selection of computation kernels


#_update_mass_ratios = operators_numba._update_mass_ratios

_update_mass_ratios = operators_numba._update_mass_ratios_para
"""
Computation of stoechiometric ratios
Parameters
----------
    mass : :class:`numpy.ndarray` of float, shape (nx,ny,nc,na)
        biomass tensor
    tmass :class:`numpy.ndarray` of float, shape (nx,ny,nc,1)
        precomputed mass summed over atom axis,
    mass_ratios :class:`numpy.ndarray` of float, shape (nx,ny,nc,na)
        out buffer to store ratios

Return
------
    mass_ratios : :class:`numpy.ndarray` of float, shape (nx,ny,nc,na)
        When tmass is 0, stoechiometry is undefined and replaced by 0
"""
#_update_biomass_fields = operators_numba._update_biomass_fields_full
_update_biomass_fields = operators_numba._update_biomass_fields_full_para

#_apply_delta_sum = operators_numba._apply_delta_sum
#_apply_delta_sum = operators_numba._apply_delta_sum_para
_apply_delta_sum = operators_numpy._apply_delta_sum
#_sub_degrad_decay_rate = operators_numba._sub_degrad_decay_rate
#_sub_degrad_decay_rate = operators_numba._sub_degrad_decay_rate_full
_sub_degrad_decay_rate = operators_numba._sub_degrad_decay_rate_full_para
#_mon_uptake_decay_rate = operators_numba._mon_uptake_decay_rate_full
_mon_uptake_decay_rate = operators_numba._mon_uptake_decay_rate_full_para
#_metabolic_process_ranked = operators_numba._metabolic_process_ranked
#_metabolic_process_ranked = operators_numba._metabolic_process_ranked_full_wrap
_metabolic_process_ranked = operators_numba._metabolic_process_ranked_full_para_wrap

#_apply_mortality = operators_numba._apply_mortality
#_apply_mortality = operators_numba._apply_mortality_para
#_apply_mortality = operators_numba._apply_mortality_full
_apply_mortality = operators_numba._apply_mortality_full_para
_apply_quantized_mortality = operators_numba._apply_quantized_mortality_full_para
_apply_mortality_stoech = operators_numba._apply_mortality_stoech_full_para
_apply_quantized_mortality_stoech = operators_numba._apply_quantized_mortality_stoech_full_para
#_remove_mass_ratio = operators_numba._remove_mass_ratio
_remove_mass_ratio = operators_numba._remove_mass_ratio_para

#_mass_ratio_removal = operators_numba._mass_ratio_removal
_mass_ratio_removal = operators_numba._mass_ratio_removal_para
#_mass_nonlocal_sharing_removal = operators_numba._mass_nonlocal_sharing_removal
_mass_nonlocal_sharing_removal = operators_numba._mass_nonlocal_sharing_removal_para

_set_shifts_filtered = operators_numba._set_shifts_filtered


_move_delta_mass = operators_numba._move_delta_mass
_move_delta_mass_withmut = operators_numba._move_delta_mass_withmut
_move_quanta_mass_to_delta_filtered = operators_numba._move_quanta_mass_to_delta_filtered
_move_quanta_mass_to_delta_filtered_taxflux = operators_numba._move_quanta_mass_to_delta_filtered_taxflux
_apply_decay = operators_numba._apply_decay

_apply_quantized_birth_mutation = operators_numba._apply_quantized_birth_mutation
_apply_quantized_birth_mutation2 = operators_numba._apply_quantized_birth_mutation2


_apply_stoechiometry_correction = operators_numba._apply_stoechiometry_full_para

# below are utility functions that generates random input/outputs arguments
# for the various operators
# they are meant for testing purposes.

def _get_update_mass_ratios_arguments(nx=1, ny=1, ntax=1, na=3):

    inargs = {'mass': np.random.uniform(size=(nx, ny, ntax, na)).astype(_default_dtype),
              'tmass': None,
              'mass_ratios': np.zeros((nx, ny, ntax, na), dtype=_default_dtype)
              }
    inargs['tmass'] = np.sum(inargs['mass'], axis=-1, keepdims=True)
    out_args = (inargs['mass_ratios'])

    return inargs, out_args


def _get_monomer_uptake_arguments(nx=1, ny=1, nmon=2, nupt=2, ntax=2):
    mon = np.random.random((nx, ny, nmon, 3)).astype(_default_dtype)
    mont = np.sum(mon, axis=-1, keepdims=True)
    monr = np.zeros_like(mon)
    monr = _update_mass_ratios(mon, mont, monr)
    mic = np.zeros((nx, ny, ntax, 3)).astype(_default_dtype)
    mic_ref = np.random.random((nx, ny, ntax, 3)).astype(_default_dtype)
    mic_ref_index = 0
    monupt = np.random.random(size=(nmon, nupt)).astype(_default_dtype)
    taxuptC = np.random.random(size=(ntax, nupt)).astype(_default_dtype)
    Vmax = np.random.random(size=(nmon, nupt)).astype(_default_dtype)
    Km = np.random.random(size=(nmon, nupt)).astype(_default_dtype)

    inargs = {'mon': mon, 'mont': mont, 'monr': monr, 'mic': mic, 'mic_ref': mic_ref,
              'mic_ref_index': mic_ref_index, 'monupt': monupt, 'taxuptC': taxuptC,
              'Vmax': Vmax, 'Km': Km
              }

    outargs = (inargs['mon'], inargs['mic'])

    return inargs, outargs


def _get_metabolic_process_ranked_arguments(nx=2, ny=2, nsrc=2,
                                           ia_src=0, ranks=[1,],
                                           ntarg=[2,]
                                           ):
    nranks = len(ranks)
    if (not isinstance(ntarg,list)):
        ntarg = [ntarg,] * len(ranks)
    assert(len(ntarg) == len(ranks))

    inargs = {
            "source": np.random.random(size=(nx, ny, nsrc, 3)).astype(_default_dtype),
            "ranked_src_costs": np.random.random(size=(nranks, nsrc, 3)).astype(_default_dtype),
            "targets": [np.zeros((nx, ny, nt, 3)) for nt in ntarg],
            "targ_src_costs": [np.random.random(size=(nsrc, nt, 3)).astype(_default_dtype) for nt in ntarg ],
            "ranks": [0, ],
            "ia_src": 0,
             }
    outargs = (inargs['source'], inargs['targets'])

    return inargs, outargs


def _get_substrate_degradation_arguments(nx=1, ny=1, nsub=2, nmon=2, nenz=2):

    sub = np.random.random(size=(nx, ny, nsub, 3)).astype(_default_dtype)
    subt = np.sum(sub, axis=-1, keepdims=True)
    subr = np.zeros_like(sub)
    subr = _update_mass_ratios(sub, subt, subr)

    enzt = 0.1*np.random.random(size=(nx, ny, nenz, 1)).astype(_default_dtype)
    mon = np.random.random(size=(nx, ny, nmon, 3)).astype(_default_dtype)

    submon_mat = np.random.choice([0.0, 1.0], size=(nsub, nmon)).astype(_default_dtype)
    Vmax = np.random.random(size=(nsub, nenz)).astype(_default_dtype)
    Km = np.random.random(size=(nsub, nenz)).astype(_default_dtype)

    lignin_indexes = (0, 0)
    cellulose_indexes = (1, 0)
    enz_C_index = 0
    lci_slope = -0.8

    inargs = {
              "sub": sub,
              "subt": subt,
              "subr": subr,
              "enz": enzt,
              "enz_C_index": enz_C_index,
              "mon": mon,
              "submon_mat": submon_mat,
              "Vmax": Vmax,
              "Km": Km,
              "lignin_indexes": lignin_indexes,
              "cellulose_indexes": cellulose_indexes,
              "lci_slope": lci_slope
              }
    outargs = (inargs['sub'], inargs['mon'])

    return inargs, outargs


def _get_mortality_arguments(nr=2, nc=2, ntax=2, na=3,
                             rseed=0
                             ):

    rng = np.random.default_rng(seed=rseed)
    mass = rng.uniform(0.001, 1.0, size=(nr, nc, ntax, na)).astype(_default_dtype)
    delta_mass = np.zeros((nr, nc, ntax,  na), dtype=_default_dtype)
    recycled_mass = delta_mass[:, :, 0, :]
    recycled_mass[()] = 0.0
    tmass = np.sum(mass, axis=-1, keepdims=True)
    rmass = np.zeros_like(mass)
    rmass = _update_mass_ratios(mass, tmass, rmass)
    ddead = np.zeros_like(mass)
    dead_mass = ddead[:, :, 0, :]
    draws = rng.uniform(size=mass.shape[:-1]).astype(_default_dtype)
    death_proba = 0.5 * np.ones((ntax,), dtype=_default_dtype)
    starvation_thresholds = 0.05 * np.ones((na,), dtype=_default_dtype)
    stoechiometry_minbounds = 0.33 * np.ones((ntax, na), dtype=_default_dtype)

    inargs = {
            "mass": mass, "rmass": rmass,
            "dead_mass": dead_mass, "recycled_mass": recycled_mass,
            "starvation_thresholds": starvation_thresholds,
            "death_proba": death_proba,
            "draws": draws,
            "stoechiometry_minbounds": stoechiometry_minbounds,
             }

    outargs = (inargs["mass"], inargs["rmass"], inargs["dead_mass"], inargs["recycled_mass"])
    return inargs, outargs
