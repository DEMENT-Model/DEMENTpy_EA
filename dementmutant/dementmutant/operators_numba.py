#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numba implementations of operators



Created on Tue Jan  4 07:44:55 2022
Copyright CNRS
@author: david.coulette@ens-lyon.fr
"""

import numpy as np
from numba import njit, prange
from numba.typed import List as numbList

from dementmutant.defaults import _default_dtype


_numb_fmath = False
_numb_cache = True

_numb_opt_loc = {'cache': _numb_cache, 'fastmath': _numb_fmath, 'parallel': False}


_numb_opt_spc_loops = {
   '_update_mass_ratios': {'cache': _numb_cache, 'fastmath': _numb_fmath, 'parallel': False},
   '_update_mass_ratios_para': {'cache': _numb_cache, 'fastmath': _numb_fmath, 'parallel': True},
   '_update_biomass_fields_full': {'cache': _numb_cache, 'fastmath': _numb_fmath, 'parallel': False},
   '_update_biomass_fields_full_para': {'cache': _numb_cache, 'fastmath': _numb_fmath, 'parallel': True},
   '_sub_degrad_decay_rate': {'cache': _numb_cache, 'fastmath': _numb_fmath, 'parallel': False},
   '_sub_degrad_decay_rate_full': {'cache': _numb_cache, 'fastmath': _numb_fmath, 'parallel': False},
   '_sub_degrad_decay_rate_full_para': {'cache': _numb_cache, 'nogil': True, 'fastmath': _numb_fmath, 'parallel': True},
   '_mon_uptake_decay_rate': {'cache': _numb_cache, 'fastmath': _numb_fmath, 'parallel': False},
   '_mon_uptake_decay_rate_full': {'cache': _numb_cache, 'fastmath': _numb_fmath, 'parallel': False},
   '_mon_uptake_decay_rate_full_para': {'cache': _numb_cache, 'fastmath': _numb_fmath, 'parallel': True},
   '_metabolic_process_ranked': {'cache': _numb_cache,'nogil':True, 'fastmath': _numb_fmath, 'parallel': False},
   '_metabolic_process_ranked_full': {'cache': _numb_cache,'nogil':True, 'fastmath': _numb_fmath, 'parallel': False},
   '_metabolic_process_ranked_full_para': {'cache': _numb_cache ,'nogil':True,'fastmath': _numb_fmath, 'parallel': True},
   '_apply_decay': {'cache': _numb_cache, 'fastmath': _numb_fmath, 'parallel': False},
   '_apply_mortality':{'cache': _numb_cache, 'fastmath': _numb_fmath, 'parallel': False},
   '_apply_mortality_stoech':{'cache': _numb_cache, 'fastmath': _numb_fmath, 'parallel': False},
   '_apply_mortality_stoech_para':{'cache': _numb_cache, 'fastmath': _numb_fmath, 'parallel': True},
   '_apply_mortality_para':{'cache': _numb_cache, 'fastmath': _numb_fmath, 'parallel': True},
   '_apply_mortality_full':{'cache': _numb_cache, 'fastmath': _numb_fmath, 'parallel': False},
   '_apply_mortality_full_para':{'cache': _numb_cache, 'fastmath': _numb_fmath, 'parallel': True},
   '_apply_mortality_stoech_full':{'cache': _numb_cache, 'fastmath': _numb_fmath, 'parallel': False},
   '_apply_mortality_stoech_full_para':{'cache': _numb_cache, 'fastmath': _numb_fmath, 'parallel': True},
   '_apply_quantized_mortality_stoech_full_para':{'cache': _numb_cache, 'fastmath': _numb_fmath, 'parallel': True},
   '_apply_quantized_mortality_full_para':{'cache': _numb_cache, 'fastmath': _numb_fmath, 'parallel': True},
   '_apply_stoechiometry_full_para':{'cache': _numb_cache, 'fastmath': _numb_fmath, 'parallel': True},
   }

def enable_parallel_spc_loops(func_list=[]):
    global _numb_opt_spc_loops
    if ([] == func_list):
        func_list = list(_numb_opt_spc_loops.keys())

    for f in func_list:
        if (_numb_opt_spc_loops[f]['parallel'] is not True):
            _numb_opt_spc_loops[f]['parallel'] = True


def display_numba_options():
    print('Local functions options')
    print(_numb_opt_loc)
    print('Global functions options')
    for k, d in _numb_opt_spc_loops.items():
        print(k)
        print(d)
        print('-'*10)


@njit(**(_numb_opt_spc_loops['_update_mass_ratios']))
def _update_mass_ratios(mass, tmass, mass_ratios):
    """
    Computation of stoechiometric ratios
    Parameters
    ----------
        mass : :class:`numpy.ndarray` of float, shape (nx,ny,nc,na)
            biomass tensor
        tmass :class:`numpy.ndarray` of float, shape (nx,ny,nc,1)
            precomputed mass summed over atom axis,
        mass_ratios :class:`numpy.ndarray` of float, shape (nx,ny,nc,na)

    Return
    ------
        mass_ratios : :class:`numpy.ndarray` of float, shape (nx,ny,nc,na)
            When tmass is 0, stoechiometry is undefined and replaced by 0
    """
    nr, nc, ncomp, na = mass.shape
    nr2, nc2, ncomp2, na2 = tmass.shape
    nr3, nc3, ncomp3, na3 = mass_ratios.shape
    assert(mass.shape == mass_ratios.shape)
    assert(mass.shape[:-1] == tmass.shape[:-1])
    assert(na2 == 1)
    assert(na == na3)
    assert(na <= 4)
    assert(na3 <= 4)

    zero_def_ratio = 0.0
    for i in range(nr):
        for j in range(nc):
            for ic in range(ncomp):
                t = tmass[i, j, ic, 0]
                if (t > 0.0):
                    mass_ratios[i, j, ic, :] = mass[i, j, ic, :] / t
                else:
                    mass_ratios[i, j, ic, :] = zero_def_ratio
    return mass_ratios


@njit(**(_numb_opt_spc_loops['_update_mass_ratios_para']))
def _update_mass_ratios_para(mass, tmass, mass_ratios):
    """
        Computation of stoechiometric ratios
        Parameters :
            - mass (nx,ny,nc,na) ndarray : biomass tensor
            - tmass (nx,ny,nc,1) array : precomputed mass summed over atom axis,
            - mass_ratios (nx,ny,nc,na) : out buffer to store ratios
        Return : mass_ratios. When tmass is 0, stoechiometry is undefined and replaced
        by a fill-in default.
    """
    nr, nc, ncomp, na = mass.shape
    nr2, nc2, ncomp2, na2 = tmass.shape
    nr3, nc3, ncomp3, na3 = mass_ratios.shape
    assert(mass.shape == mass_ratios.shape)
    assert(mass.shape[:-1] == tmass.shape[:-1])
    assert(na2 == 1)
    assert(na == na3)
    assert(na <= 4)
    assert(na3 <= 4)

    zero_def_ratio = 0.0
    for i in prange(nr):
        for j in range(nc):
            for ic in range(ncomp):
                t = tmass[i, j, ic, 0]
                if (t > 0.0):
                    mass_ratios[i, j, ic, :] = mass[i, j, ic, :] / t
                else:
                    mass_ratios[i, j, ic, :] = zero_def_ratio
    return mass_ratios


@njit(**(_numb_opt_spc_loops['_update_biomass_fields_full']))
def _update_biomass_fields_full(mass, tmass, mass_ratios):
    nr, nc, ncomp, na = mass.shape
    nr2, nc2, ncomp2, na2 = tmass.shape
    nr3, nc3, ncomp3, na3 = mass_ratios.shape
    assert(mass.shape == mass_ratios.shape)
    assert(mass.shape[:-1] == tmass.shape[:-1])
    assert(na2 == 1)
    assert(na == na3)
    assert(na <= 4)
    assert(na3 <= 4)

    for ir in range(nr):
        for ic in range(nc):
            for icomp in range(ncomp):
                ttmass = 0.0
                for ia in range(na):
                    ttmass += mass[ir, ic, icomp, ia]
                tmass[ir, ic, icomp, 0] = ttmass
                if (ttmass > 0.0):
                    for ia in range(na):
                        mass_ratios[ir, ic, icomp, ia] = mass[ir, ic, icomp, ia] / ttmass
                else:
                    mass_ratios[ir, ic, icomp, ia] = 0.0

    return tmass, mass_ratios


@njit(**(_numb_opt_spc_loops['_update_biomass_fields_full_para']))
def _update_biomass_fields_full_para(mass, tmass, mass_ratios):
    nr, nc, ncomp, na = mass.shape
    nr2, nc2, ncomp2, na2 = tmass.shape
    nr3, nc3, ncomp3, na3 = mass_ratios.shape
    assert(mass.shape == mass_ratios.shape)
    assert(mass.shape[:-1] == tmass.shape[:-1])
    assert(na2 == 1)
    assert(na == na3)

    for ir in prange(nr):
        for ic in range(nc):
            for icomp in range(ncomp):
                ttmass = 0
                for ia in range(na):
                    ttmass += mass[ir, ic, icomp, ia]
                tmass[ir, ic, icomp, 0] = ttmass
                if (ttmass > 0.0):
                    for ia in range(na):
                        mass_ratios[ir, ic, icomp, ia] = mass[ir, ic, icomp, ia] / ttmass
                else:
                    mass_ratios[ir, ic, icomp, ia] = 0.0

    return tmass, mass_ratios


@njit(cache=_numb_cache,nogil=True, fastmath=False, parallel=False)
def _apply_delta_sum(mass, delta_mass):

    nr, nc, ncomp, na = mass.shape
    assert(mass.shape == delta_mass.shape)
    assert(na <= 4)

    for ir in range(nr):
        for ic in range(nc):
            for icomp in range(ncomp):
                mass[ir, ic, icomp, :] += delta_mass[ir, ic, icomp, :]

    return mass


@njit(cache=_numb_cache, nogil=True, fastmath=False, parallel=True)
def _apply_delta_sum_para(mass, delta_mass):

    nr, nc, ncomp, na = mass.shape
    assert(mass.shape == delta_mass.shape)
    assert(na <= 4)

    for ir in prange(nr):
        for ic in range(nc):
            for icomp in range(ncomp):
                mass[ir, ic, icomp, :] += delta_mass[ir, ic, icomp, :]

    return mass


#@njit(**(_numb_opt_spc_loops['_sub_degrad_decay_rate']))
#def _sub_degrad_decay_rate(
#                           sub, subt, subr,
#                           enz, enz_C_index,
#                           mon,
#                           submon_mat,
#                           Vmax, Km,
#                           lignin_indexes, cellulose_indexes,
#                           lci_slope,
#                           ):
#    nr, nc = sub.shape[:2]
#    for i in prange(nr):
#        for j in range(nc):
#            sub[i, j, :, :], mon[i, j, :, :] = _sub_degrad_decay_rate_loc(
#                                sub[i, j, :, :], subt[i, j, :, :], subr[i, j, :, :],
#                                enz[i, j, :, :], enz_C_index,
#                                mon[i, j, :, :],
#                                submon_mat,
#                                Vmax, Km,
#                                lignin_indexes, cellulose_indexes,
#                                lci_slope
#                                )
#
#    return sub, mon
#
#
#@njit(**_numb_opt_loc)
#def _sub_degrad_decay_rate_loc(
#                               sub, subt, subr,
#                               enz, enz_C_index,
#                               mon,
#                               submon_mat,
#                               Vmax, Km,
#                               lignin_indexes, cellulose_indexes,
#                               lci_slope,
#                               ):
#    nsub, na = sub.shape
#    nsub2, nenz = Vmax.shape
#    nsub3, nmon = submon_mat.shape
#
#    dec = np.zeros((nsub, ), dtype=sub.dtype)
#    for isub in range(nsub):
#        for ienz in range(nenz):
#            den = (Km[isub, ienz] + subt[isub, 0])
#            if (den > 0.0):
#                dec[isub] += (
#                              enz[ienz, enz_C_index] * Vmax[isub, ienz] * subt[isub, 0]
#                              /
#                              den
#                              )
#    # lignin/cellulose correction
#    lci_fact = 1.0
#    lignin_t = subt[lignin_indexes[0], lignin_indexes[1]]
#    cellulose_c = sub[cellulose_indexes[0], cellulose_indexes[1]]
#    den = lignin_t + cellulose_c
#    if (den > 0.0):
#        lci_fact += lci_slope * lignin_t / den
#    dec[cellulose_indexes[0]] *= lci_fact
#    #
#    dec = np.minimum(dec, subt[:, 0])
#    for isub in range(nsub):
#        delta = dec[isub] * subr[isub, :]
#        sub[isub, :] -= delta
#        for ic in range(na):
#            if (sub[isub, ic] < 0):
#                sub[isub, ic] = 0.0
#        for imon in range(nmon):
#            mon[imon, :] += submon_mat[isub, imon] * delta
#
#    return sub, mon


@njit(**(_numb_opt_spc_loops['_sub_degrad_decay_rate_full']))
def _sub_degrad_decay_rate_full(
                           sub, subt, subr,
                           enz, enz_C_index,
                           mon,
                           submon_mat,
                           Vmax, Km,
                           lignin_indexes, cellulose_indexes,
                           lci_slope,
                           ):
    nr, nc, nsub, na = sub.shape
    nsub3, nenz = Vmax.shape
    nr2, nc2, nmon, na2 = mon.shape
    nsub2, nmon2 = submon_mat.shape

    assert(na == na2)
    assert(nsub == nsub2)
    assert(nsub == nsub3)
    assert(nmon == nmon2)
    assert(nr == nr2)
    assert(nc == nc2)
    assert(na <= 4)
    assert(na2 <= 4)

    for i in range(nr):
        for j in range(nc):
            # lignin/cellulose correction
            lci_fact = 1.0
            lignin_t = subt[i, j, lignin_indexes[0], lignin_indexes[1]]
            cellulose_c = sub[i, j, cellulose_indexes[0], cellulose_indexes[1]]
            den = lignin_t + cellulose_c
            if (den > 0):
                lci_fact += lci_slope * lignin_t / den
            for isub in range(nsub):
                dec = 0.0
                st = subt[i, j, isub, 0]
                for ienz in range(nenz):
                    den = (Km[isub, ienz] + subt[i, j, isub, 0])
                    if (den > 0.0):
                        dec += (
                            enz[i, j, ienz, enz_C_index] * Vmax[isub, ienz] * st
                            /
                            den
                            )

                if (isub == cellulose_indexes[0]):
                    dec *= lci_fact
                mdec = min(dec, st)
                delta = mdec * subr[i, j, isub, :]
                sub[i, j, isub, :] -= delta
                for ic in range(na):
                    if (sub[i, j, isub, ic] < 0):
                        sub[i, j, isub, ic] = 0.0
                for imon in range(nmon):
                    mon[i, j, imon, :] += submon_mat[isub, imon] * delta


    return sub, mon


@njit(**(_numb_opt_spc_loops['_sub_degrad_decay_rate_full_para']))
def _sub_degrad_decay_rate_full_para(
                           sub, subt, subr,
                           enz, enz_C_index,
                           mon,
                           submon_mat,
                           Vmax, Km,
                           lignin_indexes, cellulose_indexes,
                           lci_slope,
                           ):
    nr, nc, nsub, na = sub.shape
    nsub3, nenz = Vmax.shape
    nr2, nc2, nmon, na2 = mon.shape
    nsub2, nmon2 = submon_mat.shape

    assert(na == na2)
    assert(nsub == nsub2)
    assert(nsub == nsub3)
    assert(nmon == nmon2)
    assert(nr == nr2)
    assert(nc == nc2)
    assert(na <= 4)
    assert(na2 <= 4)

    for i in prange(nr):
        for j in range(nc):
            # lignin/cellulose correction
            lci_fact = 1.0
            lignin_t = subt[i, j, lignin_indexes[0], lignin_indexes[1]]
            cellulose_c = sub[i, j, cellulose_indexes[0], cellulose_indexes[1]]
            den = lignin_t + cellulose_c
            if (den > 0):
                lci_fact += lci_slope * lignin_t / den
            for isub in range(nsub):
                dec = 0.0
                st = subt[i, j, isub, 0]
                for ienz in range(nenz):
                    den = (Km[isub, ienz] + subt[i, j, isub, 0])
                    if (den > 0.0):
                        dec += (
                            enz[i, j, ienz, enz_C_index] * Vmax[isub, ienz] * st
                            /
                            den
                            )
                if (isub == cellulose_indexes[0]):
                    dec *= lci_fact
                mdec = min(dec, st)
                delta = mdec * subr[i, j, isub, :]
                sub[i, j, isub, :] -= delta
                for ic in range(na):
                    if (sub[i, j, isub, ic] < 0):
                        sub[i, j, isub, ic] = 0.0
                for imon in range(nmon):
                    mon[i, j, imon, :] += submon_mat[isub, imon] * delta

    return sub, mon


@njit(**(_numb_opt_spc_loops['_mon_uptake_decay_rate_full']))
def _mon_uptake_decay_rate_full(
                           mon, monr, mont,
                           mic, mic_ref, mic_ref_index,
                           monupt, taxuptC, Vmax, Km
                           ):
    """
        Monomer uptake computation

        Parameters:
            mon : monomer mass field (nx,ny,nmon,3)
            monr : monomer mass ratio field (nx,ny,nmon,3)
            mont : monomer total mass field (nx,ny,nmon,1)
            mic : microbial mass field
            mic_ref : microbial total reference field for uptake concentration computation
            mic_ref_index (int): index of atom component in reference field
            monupt: monomer/uptake enzyme matrix (nmon, n_uptake)
            taxpuptC : taxon uptake enzyme C constitutive production efficiency (ntax, n_uptake)
            Vmax, Km : MM kinetic parameters (nmon, nuptake)
        Return:
            - updated monomer field with decay
            - uptated microbial field with decay added
            ( can be used to simply compute uptake if mic is a zero-filled field)
    """
    nr, nc, nmon, na = mon.shape
    assert(mon.shape == monr.shape)
    assert(mon.shape[:-1] == mont.shape[:-1])
    assert(mont.shape[-1] == 1)
    nr2, nc2, ntax, na2 = mic.shape
    assert(nr == nr2)
    assert(nc == nc2)
    assert(na == na2)
    ntax2, nupt = taxuptC.shape
    assert(ntax == ntax2)
    nmon2 = mon.shape[2]
    assert(nmon == nmon2)
    assert(na <= 4)
    assert(na2 <= 4)
    for i in range(nr):
        dec = np.empty((ntax, ), dtype=mon.dtype)
        decmon = np.empty((mon.shape[-1], ), dtype=mon.dtype)
        for j in range(nc):
            for imon in range(nmon):
                dec[:] = 0.0
                locmont = mont[i, j, imon, 0]
                locmicref = mic_ref[i, j, :, mic_ref_index]
                for iupt in range(nupt):
                    den = (Km[imon, iupt] + locmont)
                    if (den > 0):
                        tmp_c = (Vmax[imon, iupt] * locmont / den)
                        dec[:] += tmp_c * taxuptC[:, iupt] * locmicref
                csmu = 0.0
                for itax in range(ntax):
                    csmu += dec[itax]
                if (csmu > 0.0):
                    rat = min(locmont, csmu) / csmu
                    dec *= rat
                    decmon[:] = 0.0
                    for itax in range(ntax):
                        delta = dec[itax] * monr[i, j, imon, :]
                        decmon += delta
                        mic[i, j, itax, :] += delta
                    mon[i, j, imon, :] -= decmon
                    for ia in range(na):
                        if (mon[i, j, imon, ia] < 0):
                            mon[i, j, imon, ia] = 0

    return mon, mic

@njit(**(_numb_opt_spc_loops['_mon_uptake_decay_rate_full_para']))
def _mon_uptake_decay_rate_full_para(
                           mon, monr, mont,
                           mic, mic_ref, mic_ref_index,
                           monupt, taxuptC, Vmax, Km
                           ):
    """
        Monomer uptake computation

        Parameters:
            mon : monomer mass field (nx,ny,nmon,3)
            monr : monomer mass ratio field (nx,ny,nmon,3)
            mont : monomer total mass field (nx,ny,nmon,1)
            mic : microbial mass field
            mic_ref : microbial total reference field for uptake concentration computation
            mic_ref_index (int): index of atom component in reference field
            monupt: monomer/uptake enzyme matrix (nmon, n_uptake)
            taxpuptC : taxon uptake enzyme C constitutive production efficiency (ntax, n_uptake)
            Vmax, Km : MM kinetic parameters (nmon, nuptake)
        Return:
            - updated monomer field with decay
            - uptated microbial field with decay added
            ( can be used to simply compute uptake if mic is a zero-filled field)
    """
    nr, nc, nmon, na = mon.shape
    assert(mon.shape == monr.shape)
    assert(mon.shape[:-1] == mont.shape[:-1])
    assert(mont.shape[-1] == 1)
    nr2, nc2, ntax, na2 = mic.shape
    assert(nr == nr2)
    assert(nc == nc2)
    assert(na == na2)
    ntax2, nupt = taxuptC.shape
    assert(ntax == ntax2)
    nmon2 = mon.shape[2]
    assert(nmon == nmon2)
    assert(na <= 4)
    assert(na2 <= 4)

    for i in prange(nr):
        dec = np.empty((ntax, ), dtype=mon.dtype)
        decmon = np.empty((na, ), dtype=mon.dtype)
        for j in range(nc):
            for imon in range(nmon):
                dec[:] = 0.0
                locmont = mont[i, j, imon, 0]
                locmicref = mic_ref[i, j, :, mic_ref_index]
                for iupt in range(nupt):
                    den = (Km[imon, iupt] + locmont)
                    if (den > 0):
                        tmp_c = (Vmax[imon, iupt] * locmont / den)
                        dec[:] += tmp_c * taxuptC[:, iupt] * locmicref
                csmu = 0.0
                for itax in range(ntax):
                    csmu += dec[itax]
                if (csmu > 0.0):
                    rat = min(locmont, csmu) / csmu
                    dec *= rat
                    decmon[:] = 0.0
                    for itax in range(ntax):
                        delta = dec[itax] * monr[i, j, imon, :]
                        decmon += delta
                        mic[i, j, itax, :] += delta
                    mon[i, j, imon, :] -= decmon
                    for ia in range(na):
                        if (mon[i, j, imon, ia] < 0):
                            mon[i, j, imon, ia] = 0
    return mon, mic

#@njit(**_numb_opt_loc)
#def _metabolic_process_ranked_loc(
#                                source, ia_src, ranked_src_costs,
#                                targets, targ_src_costs, targ_ranks,
#                                ):
#    n_rank, n_src, na_src = ranked_src_costs.shape
#    dec_src = np.zeros_like(source)
#    xi_src = np.zeros((n_rank, n_src), dtype=ranked_src_costs.dtype)
#    avail = np.zeros((source.shape[1], ), dtype=source.dtype)
#    for i_src in range(n_src):
#        avail[:] = source[i_src, :]
#        if (avail[ia_src] <= 0.0):
#            continue
#        for i_rank in range(n_rank):
#            max_dec = ranked_src_costs[i_rank, i_src, :] * source[i_src, ia_src]
#            xi_av = 1.0
#            for ia in range(max_dec.shape[0]):
#                if (max_dec[ia] > 0.0):
#                    xt = avail[ia] / max_dec[ia]
#                    xi_av = min(xt, xi_av)
#            xi_src[i_rank, i_src] = xi_av
#            eff_dec = max_dec * xi_av
#            avail -= eff_dec
#            dec_src[i_src, :] += eff_dec
#
#    for targ, targ_src_c, i_rank in zip(targets, targ_src_costs, targ_ranks):
#        for i_src in range(n_src):
#            base = xi_src[i_rank, i_src] * source[i_src, ia_src]
#            for ia in range(targ.shape[1]):
#                targ[:, ia] += targ_src_c[i_src, :, ia] * base
#
#    source -= dec_src
#
#    return source, targets
#
#
#@njit(**(_numb_opt_spc_loops['_metabolic_process_ranked']))
#def _metabolic_process_ranked_loop(
#                                 source, ia_src, ranked_src_costs,
#                                 targets, targ_src_costs, ranks,
#                                 ):
#    nr, nc = source.shape[:2]
#    for i in prange(nr):
#        for j in range(nc):
#            lctargs = [t[i, j, :, :] for t in targets]
#            source[i, j, :, :], lctargs = _metabolic_process_ranked_loc(
#                    source[i, j, :, :], ia_src, ranked_src_costs,
#                    lctargs, targ_src_costs, ranks
#                    )
#    return source, targets


@njit(**(_numb_opt_spc_loops['_metabolic_process_ranked_full']))
def _metabolic_process_ranked_full(
                                 source, ia_src, ranked_src_costs,
                                 targets, targ_src_costs, ranks,
                                 ):
    nr, nc, n_src, na_src = source.shape
    n_rank = ranked_src_costs.shape[0]

    ltarg = len(targets)
    lcosts = len(targ_src_costs)
    lranks = len(ranks)
    assert(ltarg == lcosts)
    assert(ltarg == lranks)
    dec_src = np.empty((na_src,), dtype=source.dtype)
    xi_src = np.empty((n_rank,), dtype=ranked_src_costs.dtype)
    avail = np.empty((na_src, ), dtype=source.dtype)
    for i in range(nr):
        for j in range(nc):
            for i_src in range(n_src):
                dec_src[:] = 0
                xi_src[:] = 0
                avail[:] = source[i, j, i_src, :]
                if (avail[ia_src] == 0):
                    continue
                for i_rank in range(n_rank):
                    if (avail[ia_src] <= 0.0):
                        continue
                    max_dec = ranked_src_costs[i_rank, i_src, :] * source[i, j, i_src, ia_src]
                    xi_av = 1.0
                    for ia in range(na_src):
                        if (max_dec[ia] > 0.0):
                            xt = avail[ia] / max_dec[ia]
                            xi_av = min(xt, xi_av)
                    xi_src[i_rank] = xi_av
                    eff_dec = max_dec * xi_av
                    avail -= eff_dec
                    dec_src[:] += eff_dec
                for targ, targ_src_c, i_rank in zip(targets, targ_src_costs, ranks):
                    base = xi_src[i_rank] * source[i, j, i_src, ia_src]
                    for ia in range(targ.shape[-1]):
                        targ[i, j, :, ia] += targ_src_c[i_src, :, ia] * base

                source[i, j, i_src, :] -= dec_src
    return source, targets


@njit(**(_numb_opt_spc_loops['_metabolic_process_ranked_full_para']))
def _metabolic_process_ranked_full_para(
                                 source, ia_src, ranked_src_costs,
                                 targets, targ_src_costs, ranks,
                                 ):
    nr, nc, n_src, na_src = source.shape
    n_rank = ranked_src_costs.shape[0]
    ltarg = len(targets)
    lcosts = len(targ_src_costs)
    lranks = len(ranks)
    assert(ltarg == lcosts)
    assert(ltarg == lranks)

    for i in prange(nr):
        dec_src = np.empty((na_src,), dtype=source.dtype)
        xi_src = np.empty((n_rank,), dtype=ranked_src_costs.dtype)
        avail = np.empty((na_src, ), dtype=source.dtype)
        eff_dec = np.empty((na_src, ), dtype=source.dtype)
        for j in range(nc):
            for i_src in range(n_src):
                dec_src[:] = 0
                xi_src[:] = 0
                avail[:] = source[i, j, i_src, :]
                if (avail[ia_src] <= 0):
                    continue
                for i_rank in range(n_rank):
                    max_dec = ranked_src_costs[i_rank, i_src, :] * source[i, j, i_src, ia_src]
                    xi_av = 1.0
                    for ia in range(na_src):
                        if (max_dec[ia] > 0.0):
                            xt = avail[ia] / max_dec[ia]
                            xi_av = min(xt, xi_av)
                    xi_src[i_rank] = xi_av
                    eff_dec[:] = max_dec * xi_av
                    avail -= eff_dec
                    dec_src[:] += eff_dec
                for targ, targ_src_c, i_rank in zip(targets, targ_src_costs, ranks):
                    base = xi_src[i_rank] * source[i, j, i_src, ia_src]
                    for ia in range(targ.shape[-1]):
                        targ[i, j, :, ia] += targ_src_c[i_src, :, ia] * base

                source[i, j, i_src, :] -= dec_src
    return source, targets


#def _metabolic_process_ranked(
#                                   source, ia_src, ranked_src_costs,
#                                   targets, targ_src_costs, ranks,
#                                   ):
#    targ_src_costs = numbList(targ_src_costs)
#    ranks = numbList(ranks)
#    targets = numbList(targets)
#    source, targets = _metabolic_process_ranked_loop(
#                                 source, ia_src, ranked_src_costs,
#                                 targets, targ_src_costs, ranks,
#                                 )
#    return source, targets


def _metabolic_process_ranked_full_wrap(
                                   source, ia_src, ranked_src_costs,
                                   targets, targ_src_costs, ranks,
                                   ):
    targ_src_costs = numbList(targ_src_costs)
    ranks = numbList(ranks)
    targets = numbList(targets)
    source, targets = _metabolic_process_ranked_full(
                                 source, ia_src, ranked_src_costs,
                                 targets, targ_src_costs, ranks,
                                 )
    return source, targets


def _metabolic_process_ranked_full_para_wrap(
                                   source, ia_src, ranked_src_costs,
                                   targets, targ_src_costs, ranks,
                                   ):
    targ_src_costs = numbList(targ_src_costs)
    ranks = numbList(ranks)
    targets = numbList(targets)
    source, targets = _metabolic_process_ranked_full_para(
                                 source, ia_src, ranked_src_costs,
                                 targets, targ_src_costs, ranks,
                                 )
    return source, targets


@njit(**_numb_opt_loc)
def _starvation_loc_per_taxon(mass, delta_mass, starvation_thresholds):
    starved = False
    for ia in range(mass.shape[0]):
        if (mass[ia] < starvation_thresholds[ia]):
            starved = True
            delta_mass[:] += mass[:]
            mass[:] = 0.0
    return starved, mass, delta_mass


@njit(**_numb_opt_loc)
def _stochastic_death_loc_per_taxon(mass, delta_mass, dproba, draw):
    dead = draw < dproba
    if (dead):
        delta_mass += mass[:]
        mass[:] = 0.0
    return dead, mass, delta_mass


@njit(**_numb_opt_loc)
def _stoechiometry_correction_loc_per_taxon(mass, rmass, delta_mass, rmin):
    na = mass.shape[0]
    delta_r = rmass[:]-rmin
    delta_rminus = 0.0
    delta_rplus = 0.0
    has_no_defect = True
    for ia in range(na):
        if (delta_r[ia] < 0.0):
            delta_rminus += delta_r[ia]
            has_no_defect = False
        else:
            delta_rplus += delta_r[ia]
    if (has_no_defect):
        return mass, rmass, delta_mass
    else:
        if (delta_rplus == 0.0):
            print('AAAARG', delta_r)
        scal = delta_rminus / delta_rplus
        for ia in range(na):
            if (delta_r[ia] < 0.0):
                rmass[ia] = rmin[ia]
            else:
                rmass[ia] = rmass[ia] + delta_r[ia] * scal
        i_min = np.argmin(delta_r / rmin)
        m_scal = mass[i_min] / rmin[i_min]
        d1 = m_scal * rmass
        delta_mass += mass[:]-d1[:]
        mass[:] = d1[:]
        return mass, rmass, delta_mass


@njit(**(_numb_opt_spc_loops['_apply_mortality_stoech']))
def _apply_mortality_stoech(
                     mass, rmass, dead_mass, recycled_mass,
                     starvation_thresholds,
                     death_proba,
                     draws,
                     stoechiometry_minbounds,
                     ):
    nr, nc, ntaxa, na = mass.shape
    dead_mass_loc = np.zeros((na,), dtype=_default_dtype)
    recycled_mass_loc = np.zeros((na,), dtype=_default_dtype)
    for i in range(nr):
        for j in range(nc):
            dead_mass_loc[:] = 0.0
            recycled_mass_loc[:] = 0.0
            for itax in range(ntaxa):
                locmass = mass[i, j, itax, :]
                locrmass = rmass[i, j, itax, :]
                dead, locmass, dead_mass_loc = _starvation_loc_per_taxon(locmass,
                                                                         dead_mass_loc,
                                                                         starvation_thresholds,
                                                                         )
                if (dead):
                    continue
                dead, locmass, dead_mass_loc = _stochastic_death_loc_per_taxon(locmass,
                                                                               dead_mass_loc,
                                                                               death_proba[itax],
                                                                               draws[i, j, itax]
                                                                               )
                if (dead):
                    continue
                locmass, locrmass, recycled_mass_loc = (
                        _stoechiometry_correction_loc_per_taxon(
                                                                locmass,
                                                                locrmass,
                                                                recycled_mass_loc,
                                                                stoechiometry_minbounds[itax, :],
                                                                )
                                                     )
            dead_mass[i, j, :] += dead_mass_loc[:]
            recycled_mass[i, j, :] += recycled_mass_loc[:]

    return mass, rmass, dead_mass, recycled_mass


@njit(**(_numb_opt_spc_loops['_apply_mortality']))
def _apply_mortality(
                     mass, dead_mass,
                     starvation_thresholds,
                     death_proba,
                     draws,
                     ):
    nr, nc, ntaxa, na = mass.shape
    dead_mass_loc = np.zeros((na,), dtype=_default_dtype)
    for i in range(nr):
        for j in range(nc):
            dead_mass_loc[:] = 0.0
            for itax in range(ntaxa):
                locmass = mass[i, j, itax, :]
                dead, locmass, dead_mass_loc = _starvation_loc_per_taxon(locmass,
                                                                         dead_mass_loc,
                                                                         starvation_thresholds,
                                                                         )
                if (dead):
                    continue
                dead, locmass, dead_mass_loc = _stochastic_death_loc_per_taxon(locmass,
                                                                               dead_mass_loc,
                                                                               death_proba[itax],
                                                                               draws[i, j, itax]
                                                                               )
            dead_mass[i, j, :] += dead_mass_loc[:]

    return mass, dead_mass


@njit(**(_numb_opt_spc_loops['_apply_mortality_stoech_para']))
def _apply_mortality_stoech_para(
                     mass, rmass, dead_mass, recycled_mass,
                     starvation_thresholds,
                     death_proba,
                     draws,
                     stoechiometry_minbounds,
                     ):
    nr, nc, ntaxa, na = mass.shape
    for i in prange(nr):
        dead_mass_loc = np.empty((na,), dtype=_default_dtype)
        recycled_mass_loc = np.empty((na,), dtype=_default_dtype)
        for j in range(nc):
            dead_mass_loc[:] = 0.0
            recycled_mass_loc[:] = 0.0
            for itax in range(ntaxa):
                locmass = mass[i, j, itax, :]
                locrmass = rmass[i, j, itax, :]
                dead, locmass, dead_mass_loc = _starvation_loc_per_taxon(locmass,
                                                                         dead_mass_loc,
                                                                         starvation_thresholds,
                                                                         )
                if (dead):
                    continue
                dead, locmass, dead_mass_loc = _stochastic_death_loc_per_taxon(locmass,
                                                                               dead_mass_loc,
                                                                               death_proba[itax],
                                                                               draws[i, j, itax]
                                                                               )
                if (dead):
                    continue
                locmass, locrmass, recycled_mass_loc = (
                        _stoechiometry_correction_loc_per_taxon(
                                                                locmass,
                                                                locrmass,
                                                                recycled_mass_loc,
                                                                stoechiometry_minbounds[itax, :],
                                                                )
                                                     )
            dead_mass[i, j, :] += dead_mass_loc[:]
            recycled_mass[i, j, :] += recycled_mass_loc[:]

    return mass, rmass, dead_mass, recycled_mass


@njit(**(_numb_opt_spc_loops['_apply_mortality_para']))
def _apply_mortality_para(
                     mass, dead_mass,
                     starvation_thresholds,
                     death_proba,
                     draws,
                     ):
    nr, nc, ntaxa, na = mass.shape
    for i in prange(nr):
        dead_mass_loc = np.empty((na,), dtype=_default_dtype)
        for j in range(nc):
            dead_mass_loc[:] = 0.0
            for itax in range(ntaxa):
                locmass = mass[i, j, itax, :]
                dead, locmass, dead_mass_loc = _starvation_loc_per_taxon(locmass,
                                                                         dead_mass_loc,
                                                                         starvation_thresholds,
                                                                         )
                if (dead):
                    continue
                dead, locmass, dead_mass_loc = _stochastic_death_loc_per_taxon(locmass,
                                                                               dead_mass_loc,
                                                                               death_proba[itax],
                                                                               draws[i, j, itax]
                                                                               )
            dead_mass[i, j, :] += dead_mass_loc[:]

    return mass, dead_mass


@njit(**(_numb_opt_spc_loops['_apply_mortality_stoech_full']))
def _apply_mortality_stoech_full(
                     mass, rmass, dead_mass, recycled_mass,
                     starvation_thresholds,
                     death_proba,
                     draws,
                     stoechiometry_minbounds,
                     ):
    nr, nc, ntaxa, na = mass.shape
    dead_mass_loc = np.empty((na,), dtype=mass.dtype)
    recycled_mass_loc = np.empty((na,), dtype=mass.dtype)
    delta_r = np.empty((na,), dtype=mass.dtype)
    rmin = np.empty((na,), dtype=mass.dtype)
    for i in range(nr):
        for j in range(nc):
            dead_mass_loc[:] = 0.0
            recycled_mass_loc[:] = 0.0
            for itax in range(ntaxa):
                locmass = mass[i, j, itax, :]
                locrmass = rmass[i, j, itax, :]
                rmin = stoechiometry_minbounds[itax, :]
                dead = False
                for ia in range(na):
                    if (locmass[ia] <= starvation_thresholds[ia]):
                        dead = True
                if (dead):
                    dead_mass_loc[:] += locmass[:]
                    locmass[:] = 0.0
                    locrmass[:] = 0.0
                    continue
                dead = draws[i, j, itax] <= death_proba[itax]
                if (dead):
                    dead_mass_loc += locmass[:]
                    locmass[:] = 0.0
                    locrmass[:] = 0.0
                    continue

                delta_r[:] = locrmass[:]-rmin
                delta_rminus = 0.0
                delta_rplus = 0.0
                has_no_defect = True
                for ia in range(na):
                    if (delta_r[ia] < 0.0):
                        delta_rminus += delta_r[ia]
                        has_no_defect = False
                    else:
                        delta_rplus += delta_r[ia]
                if (has_no_defect):
                    continue
                else:
                    if (delta_rplus == 0.0):
                        print('AAAARG', delta_r)
                    scal = delta_rminus / delta_rplus
                    for ia in range(na):
                        if (delta_r[ia] < 0.0):
                            locrmass[ia] = rmin[ia]
                        else:
                            locrmass[ia] = locrmass[ia] + delta_r[ia] * scal
                    i_min = np.argmin(delta_r / rmin)
                    m_scal = locmass[i_min] / rmin[i_min]
                    d1 = m_scal * locrmass
                    recycled_mass_loc += locmass[:]-d1[:]
                    locmass[:] = d1[:]

            dead_mass[i, j, :] += dead_mass_loc[:]
            recycled_mass[i, j, :] += recycled_mass_loc[:]

    return mass, rmass, dead_mass, recycled_mass


@njit(**(_numb_opt_spc_loops['_apply_mortality_full']))
def _apply_mortality_full(
                     mass, dead_mass,
                     starvation_thresholds,
                     death_proba,
                     draws,
                     ):
    nr, nc, ntaxa, na = mass.shape
    dead_mass_loc = np.empty((na,), dtype=mass.dtype)
    for i in range(nr):
        for j in range(nc):
            dead_mass_loc[:] = 0.0
            for itax in range(ntaxa):
                locmass = mass[i, j, itax, :]
                dead = False
                for ia in range(na):
                    if (locmass[ia] <= starvation_thresholds[ia]):
                        dead = True
                if (dead):
                    dead_mass_loc[:] += locmass[:]
                    locmass[:] = 0.0
                    continue
                dead = draws[i, j, itax] <= death_proba[itax]
                if (dead):
                    dead_mass_loc += locmass[:]
                    locmass[:] = 0.0
                    continue

            dead_mass[i, j, :] += dead_mass_loc[:]

    return mass, dead_mass


@njit(**(_numb_opt_spc_loops['_apply_stoechiometry_full_para']))
def _apply_stoechiometry_full_para(mass,
                                   rmass,
                                   recycled_mass,
                                   stoechiometry_minbounds):
    nr, nc, ntaxa, na = mass.shape
    nr2, nc2, ntaxa2, na2 = rmass.shape
    nr3,nc3,na3 = recycled_mass.shape
    ntaxa3, na4 = stoechiometry_minbounds.shape
    assert(nr == nr2)
    assert(nr == nr3)
    assert(nc == nc2)
    assert(nc == nc3)
    assert(na == na2)
    assert(na == na3)
    assert(na == na4)
    assert(ntaxa == ntaxa2)
    assert(ntaxa == ntaxa3)

    for ir in prange(nr):
        recycled_mass_loc = np.empty((na,), dtype=mass.dtype)
        delta_r = np.empty((na,), dtype=mass.dtype)
        rmin = np.empty((na,), dtype=mass.dtype)
        for ic in range(nc):
            recycled_mass_loc[:] = 0.0
            for itax in range(ntaxa):
                locmass = mass[ir, ic, itax, :]
                tm = np.max(locmass)
                if (tm == 0):
                    continue
                locrmass = rmass[ir, ic, itax, :]
                rmin = stoechiometry_minbounds[itax, :]
                delta_r[:] = locrmass[:]-rmin
                delta_rminus = 0.0
                delta_rplus = 0.0
                has_no_defect = True
                for ia in range(na):
                    if (delta_r[ia] < 0.0):
                        delta_rminus += delta_r[ia]
                        has_no_defect = False
                    else:
                        delta_rplus += delta_r[ia]
                if (has_no_defect):
                    continue
                else:
                    if (delta_rplus == 0.0):
                        print('BBBAAAARG', locmass, locrmass, delta_r)
                    scal = delta_rminus / delta_rplus
                    for ia in range(na):
                        if (delta_r[ia] < 0.0):
                            locrmass[ia] = rmin[ia]
                        else:
                            locrmass[ia] = locrmass[ia] + delta_r[ia] * scal
                    i_min = np.argmin(delta_r / rmin)
                    m_scal = locmass[i_min] / rmin[i_min]
                    d1 = m_scal * locrmass
                    recycled_mass_loc += locmass[:]-d1[:]
                    locmass[:] = d1[:]
                    mass[ir, ic, itax, :] = locmass
                    rmass[ir, ic, itax, :] = locrmass

            recycled_mass[ir, ic, :] += recycled_mass_loc[:]

    return mass, rmass, recycled_mass


@njit(**(_numb_opt_spc_loops['_apply_mortality_full_para']))
def _apply_mortality_full_para(
                     mass, dead_mass,
                     starvation_thresholds,
                     death_proba,
                     draws,
                     ):
    nr, nc, ntaxa, na = mass.shape
    for i in prange(nr):
        dead_mass_loc = np.empty((na,), dtype=mass.dtype)
        for j in range(nc):
            dead_mass_loc[:] = 0.0
            for itax in range(ntaxa):
                locmass = mass[i, j, itax, :]
                dead = False
                for ia in range(na):
                    if (locmass[ia] <= starvation_thresholds[ia]):
                        dead = True
                if (dead):
                    dead_mass_loc[:] += locmass[:]
                    locmass[:] = 0.0
                    continue
                dead = draws[i, j, itax] <= death_proba[itax]
                if (dead):
                    dead_mass_loc += locmass[:]
                    locmass[:] = 0.0
                    continue

            dead_mass[i, j, :] += dead_mass_loc[:]

    return mass, dead_mass


@njit(**(_numb_opt_spc_loops['_apply_mortality_stoech_full_para']))
def _apply_mortality_stoech_full_para(
                     mass, rmass, dead_mass, recycled_mass,
                     starvation_thresholds,
                     death_proba,
                     draws,
                     stoechiometry_minbounds,
                     ):
    nr, nc, ntaxa, na = mass.shape
    for i in prange(nr):
        dead_mass_loc = np.empty((na,), dtype=mass.dtype)
        recycled_mass_loc = np.empty((na,), dtype=mass.dtype)
        delta_r = np.empty((na,), dtype=mass.dtype)
        rmin = np.empty((na,), dtype=mass.dtype)
        for j in range(nc):
            dead_mass_loc[:] = 0.0
            recycled_mass_loc[:] = 0.0
            for itax in range(ntaxa):
                locmass = mass[i, j, itax, :]
                locrmass = rmass[i, j, itax, :]
                rmin = stoechiometry_minbounds[itax, :]
                dead = False
                for ia in range(na):
                    if (locmass[ia] <= starvation_thresholds[ia]):
                        dead = True
                if (dead):
                    dead_mass_loc[:] += locmass[:]
                    locmass[:] = 0.0
                    locrmass[:] = 0.0
                    continue
                dead = draws[i, j, itax] <= death_proba[itax]
                if (dead):
                    dead_mass_loc += locmass[:]
                    locmass[:] = 0.0
                    locrmass[:] = 0.0
                    continue

                delta_r[:] = locrmass[:]-rmin
                delta_rminus = 0.0
                delta_rplus = 0.0
                has_no_defect = True
                for ia in range(na):
                    if (delta_r[ia] < 0.0):
                        delta_rminus += delta_r[ia]
                        has_no_defect = False
                    else:
                        delta_rplus += delta_r[ia]
                if (has_no_defect):
                    continue
                else:
                    if (delta_rplus == 0.0):
                        print('AAAARG', locmass, locrmass, delta_r)
                    scal = delta_rminus / delta_rplus
                    for ia in range(na):
                        if (delta_r[ia] < 0.0):
                            locrmass[ia] = rmin[ia]
                        else:
                            locrmass[ia] = locrmass[ia] + delta_r[ia] * scal
                    i_min = np.argmin(delta_r / rmin)
                    m_scal = locmass[i_min] / rmin[i_min]
                    d1 = m_scal * locrmass
                    recycled_mass_loc += locmass[:]-d1[:]
                    locmass[:] = d1[:]

                    mass[i, j, itax, :] = locmass
                    rmass[i, j, itax, :] = locrmass

            dead_mass[i, j, :] += dead_mass_loc[:]
            recycled_mass[i, j, :] += recycled_mass_loc[:]

    return mass, rmass, dead_mass, recycled_mass


@njit(**(_numb_opt_spc_loops['_apply_quantized_mortality_stoech_full_para']))
def _apply_quantized_mortality_stoech_full_para(
                     mass, rmass, dead_mass, recycled_mass,
                     quant,
                     starvation_thresholds,
                     death_proba,
                     draws,
                     stoechiometry_minbounds,
                     ):
    nr, nc, ntaxa, na = mass.shape
    for ir in prange(nr):
        dead_mass_loc = np.empty((na,), dtype=mass.dtype)
        recycled_mass_loc = np.empty((na,), dtype=mass.dtype)
        delta_r = np.empty((na,), dtype=mass.dtype)
        rmin = np.empty((na,), dtype=mass.dtype)
        for ic in range(nc):
            dead_mass_loc[:] = 0.0
            recycled_mass_loc[:] = 0.0
            for itax in range(ntaxa):
                locmass = mass[ir, ic, itax, :]
                locrmass = rmass[ir, ic, itax, :]
                rmin = stoechiometry_minbounds[itax, :]
                all_dead = False
                nq = quant[ir,ic,itax,0]
                for ia in range(na):
                    if (locmass[ia] <= nq * starvation_thresholds[ia]):
                        all_dead = True
                if (all_dead):
                    dead_mass_loc[:] += locmass[:]
                    locmass[:] = 0.0
                    locrmass[:] = 0.0
                    quant[ir,ic,itax,0] = 0.0
                    continue
                nquant = int(quant[ir, ic, itax, 0])
                if (nquant == 0):
                    continue
                ndead = 0.0  # float to avoid cast
                for iind in range(nquant):
                    if (draws[ir, ic, itax, iind] <= death_proba[itax]):
                        ndead += 1.0
                if (ndead > 0):
                    dm = locmass[:] * (ndead / quant[ir, ic, itax, 0])
                    dead_mass_loc += dm[:]
                    locmass[:] -= dm[:]
                    quant[ir,ic,itax,0] -= ndead
                    # locrmass no uptated : stoechiometry stays constant
                if (nquant == ndead):
                    continue

                delta_r[:] = locrmass[:]-rmin
                delta_rminus = 0.0
                delta_rplus = 0.0
                has_no_defect = True
                for ia in range(na):
                    if (delta_r[ia] < 0.0):
                        delta_rminus += delta_r[ia]
                        has_no_defect = False
                    else:
                        delta_rplus += delta_r[ia]
                if (has_no_defect):
                    continue
                else:
                    if (delta_rplus == 0.0):
                        print('AAAARG', locmass, locrmass, delta_r)
                    scal = delta_rminus / delta_rplus
                    for ia in range(na):
                        if (delta_r[ia] < 0.0):
                            locrmass[ia] = rmin[ia]
                        else:
                            locrmass[ia] = locrmass[ia] + delta_r[ia] * scal
                    i_min = np.argmin(delta_r / rmin)
                    m_scal = locmass[i_min] / rmin[i_min]
                    d1 = m_scal * locrmass
                    recycled_mass_loc += locmass[:]-d1[:]
                    locmass[:] = d1[:]

                    mass[ir, ic, itax, :] = locmass
                    rmass[ir, ic, itax, :] = locrmass

            dead_mass[ir, ic, :] += dead_mass_loc[:]
            recycled_mass[ir, ic, :] += recycled_mass_loc[:]

    return mass, rmass,quant, dead_mass, recycled_mass



@njit(**(_numb_opt_spc_loops['_apply_quantized_mortality_full_para']))
def _apply_quantized_mortality_full_para(
                     mass, quant, dead_mass,
                     starvation_thresholds,
                     death_proba,
                     draws,
                     ):
    nr, nc, ntaxa, na = mass.shape
    for ir in prange(nr):
        dead_mass_loc = np.empty((na,), dtype=mass.dtype)
        for ic in range(nc):
            dead_mass_loc[:] = 0.0
            for itax in range(ntaxa):
                locmass = mass[ir, ic, itax, :]
                all_dead = False
                nq = quant[ir,ic,itax,0]
                for ia in range(na):
                    if (locmass[ia] <= nq * starvation_thresholds[ia]):
                        all_dead = True
                if (all_dead):
                    dead_mass_loc[:] += locmass[:]
                    locmass[:] = 0.0
                    quant[ir,ic,itax,0] = 0.0
                    continue
                nquant = int(quant[ir, ic, itax, 0])
                if (nquant == 0):
                    continue
                ndead = 0.0  # float to avoid cast
                for iind in range(nquant):
                    if (draws[ir, ic, itax, iind] <= death_proba[itax]):
                        ndead += 1.0
                if (ndead > 0):
                    dm = locmass[:] * (ndead / quant[ir, ic, itax, 0])
                    dead_mass_loc += dm[:]
                    locmass[:] -= dm[:]
                    quant[ir,ic,itax,0] -= ndead
                    # locrmass no uptated : stoechiometry stays constant
                if (nquant == ndead):
                    continue

            dead_mass[ir, ic, :] += dead_mass_loc[:]

    return mass,quant, dead_mass




@njit(cache=_numb_cache, nogil=True, fastmath=True, parallel=False)
def _remove_mass_ratio(mass, dmass, filt, ratio):

    nr, nc, ntax, na = mass.shape
    nr2, nc2, ntax2, na2 = dmass.shape
    nr3, nc3, ntax3, na3 = filt.shape
    assert(mass.shape == dmass.shape)
    assert(mass.shape == filt.shape)

    dm = np.empty((na,), dtype=mass.dtype)
    for ir in range(nr):
        for ic in range(nc):
            for itax in range(ntax):
                if (filt[ir, ic, itax, 0]):
                    dm = ratio * mass[ir, ic, itax, :]
                    mass[ir, ic, itax, :] -= dm
                    dmass[ir, ic, itax, :] = dm
    return mass, dmass


@njit(cache=_numb_cache, nogil=True, fastmath=True, parallel=True)
def _remove_mass_ratio_para(mass, dmass, filt, ratio):

    nr, nc, ntax, na = mass.shape
    nr2, nc2, ntax2, na2 = dmass.shape
    nr3, nc3, ntax3, na3 = filt.shape
    assert(mass.shape == dmass.shape)
    assert(mass.shape == filt.shape)
    for ir in prange(nr):
        dm = np.empty((na,), dtype=mass.dtype)
        for ic in range(nc):
            for itax in range(ntax):
                if (filt[ir, ic, itax, 0]):
                    dm = ratio * mass[ir, ic, itax, :]
                    mass[ir, ic, itax, :] -= dm
                    dmass[ir, ic, itax, :] = dm
    return mass, dmass


@njit(cache=_numb_cache, nogil=True, fastmath=False, parallel=False)
def _mass_ratio_removal(mass, dmass, fdisp,
                        tax_filt, thresholds, ratio
                        ):
    nr, nc, ntax, na = mass.shape
    nr2, nc2, ntax2, na2 = dmass.shape
    assert(mass.shape == dmass.shape)
    assert(mass.shape[:-1] == fdisp.shape)
    assert(tax_filt.shape[0] == ntax)
    assert(tax_filt.shape == thresholds.shape)

    ndisp = 0
    dm = np.empty((na,), dtype=mass.dtype)
    for ir in range(nr):
        for ic in range(nc):
            for itax in range(ntax):
                if (tax_filt[itax]) and (mass[ir, ic, itax, 0] > thresholds[itax]):
                    fdisp[ir, ic, itax] = True
                    dm[:] = ratio * mass[ir, ic, itax, :]
                    dmass[ir, ic, itax, :] = dm
                    mass[ir, ic, itax, :] -= dm
                    ndisp += 1
    return mass, dmass, fdisp, ndisp


@njit(cache=_numb_cache, nogil=True, fastmath=False, parallel=True)
def _mass_ratio_removal_para(mass, dmass, fdisp,
                             tax_filt, thresholds, ratio
                             ):

    nr, nc, ntax, na = mass.shape
    nr2, nc2, ntax2, na2 = dmass.shape
    assert(mass.shape == dmass.shape)
    assert(mass.shape[:-1] == fdisp.shape)
    assert(tax_filt.shape[0] == ntax)
    assert(tax_filt.shape == thresholds.shape)

    ndisp = 0
    for ir in prange(nr):
        dm = np.empty((na,), dtype=mass.dtype)
        locndisp = 0
        for ic in range(nc):
            for itax in range(ntax):
                if (tax_filt[itax]) and (mass[ir, ic, itax, 0] > thresholds[itax]):
                    fdisp[ir, ic, itax] = True
                    dm[:] = ratio * mass[ir, ic, itax, :]
                    dmass[ir, ic, itax, :] = dm
                    mass[ir, ic, itax, :] -= dm
                    locndisp += 1
        ndisp += locndisp
    return mass, dmass, fdisp, ndisp


@njit(cache=_numb_cache, nogil=True, fastmath=False, parallel=False)
def _mass_nonlocal_sharing_removal(mass, dmass, fdisp,
                                   tax_filt, thresholds
                                   ):

    nr, nc, ntax, na = mass.shape
    nr2, nc2, ntax2, na2 = dmass.shape
    assert(mass.shape == dmass.shape)
    assert(mass.shape[:-1] == fdisp.shape)
    assert(tax_filt.shape[0] == ntax)
    assert(tax_filt.shape == thresholds.shape)

    ndisp = 0
    nlive = 0
    mtot = np.zeros((na,), dtype=mass.dtype)
    mquant = np.zeros((na,), dtype=mass.dtype)
    for ir in range(nr):
        for ic in range(nc):
            for itax in range(ntax):
                if (tax_filt[itax]):
                    if (mass[ir, ic, itax, 0] > thresholds[itax]):
                        fdisp[ir, ic, itax] = True
                        nlive += 1
                        ndisp += 1
                        mtot += mass[ir, ic, itax, :]
                    else:
                        for ia in range(na):
                            if mass[ir, ic, itax, ia] > 0.0:
                                nlive += 1
                                mtot += mass[ir, ic, itax, :]
                                break

    mquant[:] = mtot / (nlive + ndisp)

    for ir in range(nr):
        for ic in range(nc):
            for itax in range(ntax):
                if (fdisp[ir, ic, itax]):
                    dmass[ir, ic, itax, :] = mquant[:]

    for ir in range(nr):
        for ic in range(nc):
            for itax in range(ntax):
                if (tax_filt[itax]):
                    live = False
                    for ia in range(na):
                        if mass[ir, ic, itax, ia] > 0.0:
                            live = True
                            break
                    if (live):
                        mass[ir, ic, itax, :] = mquant[:]

    return mass, dmass, fdisp, ndisp


@njit(cache=_numb_cache, nogil=True, fastmath=False, parallel=True)
def _mass_nonlocal_sharing_removal_para(mass, dmass, fdisp,
                                        tax_filt, thresholds
                                        ):

    nr, nc, ntax, na = mass.shape
    nr2, nc2, ntax2, na2 = dmass.shape
    assert(mass.shape == dmass.shape)
    assert(mass.shape[:-1] == fdisp.shape)
    assert(tax_filt.shape[0] == ntax)
    assert(tax_filt.shape == thresholds.shape)

    mtot = np.zeros((na,), dtype=mass.dtype)
    mquant = np.zeros((na,), dtype=mass.dtype)
    counts = np.zeros((2,), dtype=np.int64)

    c1 = np.array([0, 1], dtype=np.int64)

    for ir in prange(nr):
        for ic in range(nc):
            for itax in range(ntax):
                if (tax_filt[itax]):
                    if (mass[ir, ic, itax, 0] > thresholds[itax]):
                        fdisp[ir, ic, itax] = True
                        counts += 1
                        mtot += mass[ir, ic, itax, :]
                    else:
                        for ia in range(na):
                            if mass[ir, ic, itax, ia] > 0.0:
                                mtot += mass[ir, ic, itax, :]
                                counts += c1
                                break

    ndisp, nlive = counts[0], counts[1]

    mquant[:] = mtot / (nlive + ndisp)

    ndispchk = 0
    for ir in prange(nr):
        ll = 0
        for ic in range(nc):
            for itax in range(ntax):
                if (fdisp[ir, ic, itax]):
                    dmass[ir, ic, itax, :] = mquant[:]
                    ll += 1
        ndispchk += ll
    assert(ndisp == ndispchk)
    for ir in prange(nr):
        for ic in range(nc):
            for itax in range(ntax):
                if (tax_filt[itax]):
                    live = False
                    for ia in range(na):
                        if mass[ir, ic, itax, ia] > 0.0:
                            live = True
                            break
                    if (live):
                        mass[ir, ic, itax, :] = mquant[:]

    return mass, dmass, fdisp, ndisp

#@njit(nogil=True, fastmath=True, parallel=False)
#def _set_shifts_filtered(shifts, filt, draws, idim):
#    ndim, nr, nc, ntax = shifts.shape
#    nr2, nc2, ntax2, na = filt.shape
#    ndraw = draws.size
#    assert(nr == nr2)
#    assert(nc == nc2)
#    assert(ntax == ntax2)
#    assert(ndim == 2)
#    assert(idim < ndim)
#
#    count = 0
#    for ir in range(nr):
#        for ic in range(nc):
#            for itax in range(ntax):
#                if (filt[ir, ic, itax, 0]):
#                    shifts[idim, ir, ic, itax] = draws[count]
#                    count += 1
#    assert(count == ndraw)
#    return shifts


@njit(cache=_numb_cache, nogil=True, fastmath=True, parallel=False)
def _set_shifts_filtered(shifts, filt, draws, idim):
    ndim, nr, nc, ntax = shifts.shape
    nr2, nc2, ntax2 = filt.shape
    ndraw = draws.size
    assert(nr == nr2)
    assert(nc == nc2)
    assert(ntax == ntax2)
    assert(ndim == 2)
    assert(idim < ndim)

    count = 0
    for ir in range(nr):
        for ic in range(nc):
            for itax in range(ntax):
                if (filt[ir, ic, itax]):
                    shifts[idim, ir, ic, itax] = draws[count]
                    count += 1
    assert(count == ndraw)
    return shifts


@njit(cache=_numb_cache, fastmath=True)
def _move_delta_mass(mass, delta_mass, shifts):
    nr, nc, ntax, na = mass.shape
    assert(mass.shape == delta_mass.shape)
    assert(mass.shape[:-1] == shifts.shape[1:])

    for ir in range(nr):
        for ic in range(nc):
            for itax in range(ntax):
                ir1 = (ir + shifts[0, ir, ic, itax]) % nr
                ic1 = (ic + shifts[1, ir, ic, itax]) % nc
                mass[ir1, ic1, itax, :] += delta_mass[ir, ic, itax, :]
                delta_mass[ir, ic, itax, :] = 0
    return mass


@njit(cache=_numb_cache, fastmath=True)
def _move_delta_mass_withmut(mass, delta_mass, spc_shifts, tax_shifts):
    nr, nc, ntax, na = mass.shape
    assert(mass.shape == delta_mass.shape)
    assert(mass.shape[:-1] == spc_shifts.shape[1:])
    assert(tax_shifts.shape[0] == nr)
    assert(tax_shifts.shape[1] == nc)
    assert(tax_shifts.shape[2] == ntax)

    for ir in range(nr):
        for ic in range(nc):
            for itax in range(ntax):
                ir1 = (ir + spc_shifts[0, ir, ic, itax]) % nr
                ic1 = (ic + spc_shifts[1, ir, ic, itax]) % nc
                itax2 = tax_shifts[ir, ic, itax]
                mass[ir1, ic1, itax2, :] += delta_mass[ir, ic, itax, :]
                delta_mass[ir, ic, itax, :] = 0
    return mass



@njit(cache=True, parallel=False, fastmath=False)
def _move_quanta_mass_to_delta_filtered(mass,
                              dmass,
                              quant,
                              dquant,
                              filt,
                              shifts
                              ):
    nr,nc,nt,na = mass.shape
    nr2,nc2,nt2,na2= dmass.shape
    nr3,nc3,nt3,nq = quant.shape
    nr4,nc4,nt4,nq2 = dquant.shape
    ndraws, nd = shifts.shape
    nt5 = filt.shape
    assert(nr == nr2)
    assert(nr == nr3)
    assert(nr == nr4)
    assert(nc == nc2)
    assert(nc == nc3)
    assert(nc == nc4)
    assert(nt == nt2)
    assert(nt == nt3)
    assert(nt == nt4)
    assert(nq == 1)
    assert(nq2 == 1)
    assert(nd == 2)

    count = 0
    nmove = 0

    dm = np.empty((na,),dtype=mass.dtype)

    for ir in range(nr):
        for ic in range(nc):
            for itax in range(nt):
                if (not filt[itax]):
                    continue
                nqm = int(quant[ir,ic,itax,0])
                if (nqm == 0):
                    continue
                dm[:] = mass[ir,ic,itax,:] / quant[ir,ic,itax,0]
                for iq in range(nqm):
                    sr,sc = shifts[count,0], shifts[count,1]
                    ds2 = sr*sr + sc*sc
                    if (ds2 < 1):
                        count +=1
                        continue
                    else:
                        ir2 = (ir + sr) % nr
                        ic2 = (ic + sc) % nc
                        mass[ir,ic,itax,:] -= dm
                        quant[ir,ic,itax,0] -= 1.0
                        dmass[ir2,ic2,itax,:] +=dm
                        dquant[ir2,ic2,itax,0] += 1.0
                        count +=1
                        nmove +=1
    assert(count == ndraws)

    return mass,dmass,quant,dquant, nmove

@njit(cache=True, parallel=False, fastmath=False)
def _move_quanta_mass_to_delta_filtered_taxflux(mass,
                              dmass,
                              quant,
                              dquant,
                              filt,
                              shifts,
                              tax_shifts,
                              ):
    nr,nc,nt,na = mass.shape
    nr2,nc2,nt2,na2= dmass.shape
    nr3,nc3,nt3,nq = quant.shape
    nr4,nc4,nt4,nq2 = dquant.shape
    ndraws, nd = shifts.shape
    ndraws2 = tax_shifts.shape[0]
    nt5 = filt.shape
    assert(nr == nr2)
    assert(nr == nr3)
    assert(nr == nr4)
    assert(nc == nc2)
    assert(nc == nc3)
    assert(nc == nc4)
    assert(nt == nt2)
    assert(nt == nt3)
    assert(nt == nt4)
    assert(nq == 1)
    assert(nq2 == 1)
    assert(nd == 2)
    assert(ndraws == ndraws2)

    count = 0
    nmove = 0
    nbdcross = 0
    nbdchange= 0

    dm = np.empty((na,),dtype=mass.dtype)

    for ir in range(nr):
        for ic in range(nc):
            for itax in range(nt):
                if (not filt[itax]):
                    continue
                nqm = int(quant[ir,ic,itax,0])
                if (nqm == 0):
                    continue
                dm[:] = mass[ir,ic,itax,:] / quant[ir,ic,itax,0]
                for iq in range(nqm):
                    sr,sc = shifts[count,0], shifts[count,1]
                    ds2 = sr*sr + sc*sc
                    if (ds2 < 1):
                        count +=1
                        continue
                    else:
                        tmp_ir2 = ir + sr
                        tmp_ic2 = ic + sc
                        ir2 = tmp_ir2 % nr
                        ic2 = tmp_ic2 % nc
                        bd_check = (tmp_ir2 != ir2) or (tmp_ic2 != ic2)
                        if (bd_check):
                            itax2 = tax_shifts[count]
                            nbdcross +=1
                            if (itax2 != itax):
                                nbdchange +=1
                        else:
                            itax2 = itax
                        mass[ir,ic,itax,:] -= dm
                        quant[ir,ic,itax,0] -= 1.0
                        dmass[ir2,ic2,itax2,:] +=dm
                        dquant[ir2,ic2,itax2,0] += 1.0
                        count +=1
                        nmove +=1
    assert(count == ndraws)

    return mass,dmass,quant,dquant, nmove, nbdcross, nbdchange



@njit(**(_numb_opt_spc_loops['_apply_decay']))
def _apply_decay(src, targ, r):
    nr, nc, nco, na = src.shape
    for ir in prange(nr):
        for ic in range(nc):
            for ico in range(nco):
                for ia in range(na):
                    dec = r[ico] * src[ir, ic, ico, ia]
                    targ[ir, ic, ico, ia] += dec
                    src[ir, ic, ico, ia] -= dec
    return src, targ



@njit(cache=True, fastmath=False)
def _apply_quantized_birth_mutation(mass,
                          dmass,
                          quant,
                          dquant,
                          taxshifts,
                          ):
    nr,nc,ntax,na = mass.shape
    mq = np.empty((ntax,na))
    nnew = np.empty((ntax,))
    nmut = 0
    for ir in range(nr):
        for ic in range(nc):
            nnew[:] = quant[ir,ic,:,0]
            mq[()] = 0
            for itax in range(ntax):
                if (quant[ir,ic,itax,0] > 0):
                    mq[itax,:] = mass[ir,ic,itax,:] / nnew[itax]
            for itax in range(ntax):
                dn = int(dquant[ir,ic,itax,0])
                if (dn == 0):
                    continue
                for iind in range(dn):
                    itax_end = taxshifts[ir,ic,itax,iind]
                    mass[ir,ic,itax,:] -= mq[itax,:]
                    dmass[ir,ic,itax_end,:] += mq[itax,:]
                    nnew[itax] = nnew[itax] -1
                    nnew[itax_end] = nnew[itax_end]+1
                    if (itax != itax_end):
                        nmut +=1
            quant[ir,ic,:,0] = nnew[:]
#
    return mass,dmass,quant,nmut


@njit(cache=True, fastmath=False)
def _apply_quantized_birth_mutation2(mass,
                          dmass,
                          quant,
                          dquant,
                          taxshifts,
                          ):
    nr,nc,ntax,na = mass.shape
    mq = np.empty((ntax,na))
    nnew = np.empty((ntax,))
    nmut = 0
    for ir in range(nr):
        for ic in range(nc):
            nnew[:] = quant[ir,ic,:,0]
            mq[()] = 0
            for itax in range(ntax):
                if (quant[ir,ic,itax,0] > 0):
                    mq[itax,:] = 2 * mass[ir,ic,itax,:] / nnew[itax]
            for itax in range(ntax):
                dn = int(dquant[ir,ic,itax,0])
                if (dn == 0):
                    continue
                for iind in range(dn):
                    itax_end = taxshifts[ir,ic,itax,iind]
                    mass[ir,ic,itax,:] -= mq[itax,:]
                    dmass[ir,ic,itax_end,:] += mq[itax,:]
                    nnew[itax] = nnew[itax] - 2
                    nnew[itax_end] = nnew[itax_end] + 2
                    if (itax != itax_end):
                        nmut +=1
            quant[ir,ic,:,0] = nnew[:]
#
    return mass,dmass,quant,nmut

#def _sub_degrad_dummy():
#
#    nx = 1
#    ny = 1
#    nsub = 2
#    nmon = 2
#    nenz = 2
#    lignin_indexes = (0, 0)
#    cellulose_indexes = (1, 0)
#    enz_C_index = 0
#    lci_slope = -0.8
#
#    sub = np.random.random(size=(nx, ny, nsub, 3)).astype(_default_dtype)
#    subt = np.sum(sub, axis=-1, keepdims=True)
#    subr = np.zeros_like(sub)
#    subr = _update_mass_ratios(sub, subt, subr)
#
#    enzt = np.random.random(size=(nx, ny, nenz, 1)).astype(_default_dtype)
#    mon = np.random.random(size=(nx, ny, nmon, 3)).astype(_default_dtype)
#
#    submon_mat = np.random.choice([0.0, 1.0], size=(nsub, nmon)).astype(_default_dtype)
#    Vmax = np.random.random(size=(nsub, nenz)).astype(_default_dtype)
#    Km = np.random.random(size=(nsub, nenz)).astype(_default_dtype)
#
#    sub, mon = _sub_degrad_decay_rate(
#                                      sub, subt, subr,
#                                      enzt, enz_C_index,
#                                      mon,
#                                      submon_mat,
#                                      Vmax, Km,
#                                      lignin_indexes, cellulose_indexes,
#                                      lci_slope,
#                                      )


#def _mon_uptake_dummy():
#    nx = 2
#    ny = 2
#    nmon = 2
#    nupt = 2
#    ntax = 2
#
#    mon = np.random.random((nx, ny, nmon, 3)).astype(_default_dtype)
#    mont = np.sum(mon, axis=-1, keepdims=True)
#    monr = np.zeros_like(mon)
#    monr = _update_mass_ratios(mon, mont, monr)
#
#    mic = np.zeros((nx, ny, ntax, 3)).astype(_default_dtype)
#    mic_ref = np.random.random((nx, ny, ntax, 3)).astype(_default_dtype)
#
#    monupt = np.random.random(size=(nmon, nupt)).astype(_default_dtype)
#
#    taxuptC = np.random.random(size=(ntax, nupt)).astype(_default_dtype)
#
#    Vmax = np.random.random(size=(nmon, nupt)).astype(_default_dtype)
#    Km = np.random.random(size=(nmon, nupt)).astype(_default_dtype)
#
#    mon, mic = _mon_uptake_decay_rate_full(
#                                     mon, monr, mont,
#                                     mic, mic_ref, 0,
#                                     monupt, taxuptC, Vmax, Km
#                                     )


#def _apply_decay_dummy():
#    nr, nc, nco, na = 2, 2, 2, 3
#    src = np.ones((nr, nc, nco, na), dtype=_default_dtype)
#    targ = np.ones_like(src)
#    r = 0.3 * np.ones((nco,), dtype=_default_dtype)
#    src, targ = _apply_decay(src, targ, r)
#
#
#def _metabolic_process_dummy():
#    nx = 2
#    ny = 2
#    nsrc = 2
#    ia_src = 0
#    nranks = 1
#
#    ntarg = 2
#
#    source = np.random.random(size=(nx, ny, nsrc, 3)).astype(_default_dtype)
#
#    ranked_src_costs = np.random.random(size=(nranks, nsrc, 3)).astype(_default_dtype)
#
#    targets = numbList([np.zeros((nx, ny, ntarg, 3)), ])
#
#    targ_src_costs = numbList([np.random.random(size=(nsrc, ntarg, 3)).astype(_default_dtype), ])
#
#    ranks = numbList([0, ])
#
#    source, targets = _metabolic_process_ranked(
#                                 source, ia_src, ranked_src_costs,
#                                 targets, targ_src_costs, ranks,
#                                 )


#def _mortality_dummy():
#    rng = np.random.default_rng(seed=0)
#    nr, nc, ntax, na = 2, 2, 100, 3
#    np.random.seed(0)
#    mass = rng.uniform(0.001, 1.0, size=(nr, nc, ntax, na)).astype(_default_dtype)
#    delta_mass = np.zeros((nr, nc, ntax,  na), dtype=_default_dtype)
#    recycled_mass = delta_mass[:, :, 0, :]
#    recycled_mass[()] = 0.0
#    tmass = np.sum(mass, axis=-1, keepdims=True)
#    rmass = np.zeros_like(mass)
#    rmass = _update_mass_ratios(mass, tmass, rmass)
#    ddead = np.zeros_like(mass)
#    dead_mass = ddead[:, :, 0, :]
#    draws = rng.uniform(size=mass.shape[:-1]).astype(_default_dtype)
#    death_proba = 0.5 * np.ones((ntax,), dtype=_default_dtype)
#    starvation_thresholds = 0.05 * np.ones((na,), dtype=_default_dtype)
#    stoech_quotas = 0.33 * np.ones((ntax, na), dtype=_default_dtype)
##    print('BBB')
##    for vr in [mass, draws, death_proba, stoech_quotas,starvation_thresholds]:
##        print(type(vr),vr.shape, vr.dtype)
##    # special cases to ensure all subfunctions are called
##    locmass = np.ones((mass.shape[-1],))
##    dead_mass_loc = np.zeros_like(locmass)
##    dead, locmass, dead_mass_loc = _starvation_loc_per_taxon(locmass,
##                                                             dead_mass_loc,
##                                                             starvation_thresholds,
##                                                             )
##    locmass = np.ones((mass.shape[-1],))
##    dead_mass_loc = np.zeros_like(locmass)
##    dead, locmass, dead_mass_loc = _stochastic_death_loc_per_taxon(locmass,
##                                                                   dead_mass_loc,
##                                                                   0.5,
##                                                                   0.1
##                                                                   )
##    locmass = np.ones((mass.shape[-1],))
##    locrmass = locmass / np.sum(locmass)
##    recycled_mass_loc = np.zeros_like(locmass)
##    locmass, locrmass, recycled_mass_loc = (
##            _stoechiometry_correction_loc_per_taxon(
##                                                    locmass,
##                                                    locrmass,
##                                                    recycled_mass_loc,
##                                                    stoechiometry_minbounds[0, :],
##                                                    )
##            )
#
#    mass, rmass, dead_mass, recycled_mass = _apply_mortality(
#                                                    mass, rmass, dead_mass, recycled_mass,
#                                                    starvation_thresholds,
#                                                    death_proba,
#                                                    draws,
#                                                    stoech_quotas,
#                                                    )


#def _dispersal_dummy():
#    nr, nc, ntax, na = 2, 2, 2, 3
#    mass = np.ones((nr, nc, ntax, na), dtype=_default_dtype)
#    delta_mass = np.ones((nr, nc, ntax, na), dtype=_default_dtype)
#    shifts = 2 * np.ones((2, nr, nc, ntax), dtype=int)
#    mass = _move_delta_mass(mass, delta_mass, shifts)
#
#
#def _update_mass_ratio_dummy():
#    nx, ny, ntax, na = 2, 2, 2, 2
#    mass = np.random.uniform(size=(nx, ny, ntax, na)).astype(_default_dtype)
#    tmass = np.sum(mass, axis=-1, keepdims=True)
#    mass_ratios = np.zeros_like(mass)
#    mass_ratios = _update_mass_ratios(mass, tmass, mass_ratios)


#def _run_dummys_for_precompilation():
#    """
#        Call all numba jitted function with small dummy arguments
#        to trigger compilation
#        Run this before simulation if you want to avoid including
#        compilation times in simulation time measurements
#    """
#    st = StageTimer()
#    _update_mass_ratio_dummy()
#    st.tag_event('BiomassField Ratio')
#    _sub_degrad_dummy()
#    st.tag_event('Substrate degradation')
#    _mon_uptake_dummy()
#    st.tag_event('Monomer Uptake')
#    _metabolic_process_dummy()
#    st.tag_event('Metabolic process')
#    _mortality_dummy()
#    st.tag_event('Mortality')
#    _dispersal_dummy()
#    st.tag_event('Dispersal')
#    _apply_decay_dummy()
#    st.tag_event('Linear Decay')
#    st.set_end()
#    st.display()
