#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various functions for setting up and running a simulation

Created on Thu Dec 16 21:57:32 2021

Copyright CNRS

@author: david.coulette@ens-lyon.fr



"""

import sys
import os
import itertools
import numpy as np
import pandas as pd
import io
import psutil

sys.path.append('../../')  # adapt to point to the root location of dementmutant directory

import dementmutant.default_data as default_data

# from examples.commonutils.simutils import setup_diagcollector_example
from examples.commonutils.simutils import get_default_empty_ecosystem
from examples.commonutils.simutils import run_simulation_time_loop
from examples.commonutils.simutils import set_run_seed_in_directory
from examples.commonutils.simutils import load_simu_from_dir
from examples.commonutils.simutils import reload_fields_state
from examples.commonutils.simutils import get_simulation_files
from examples.commonutils.simutils import dump_diagcollector
from examples.commonutils.simutils import create_simulation_directory

from dementmutant.microbe import set_microbe_quantization_on, get_microbe_quantization












def microbe_pop_on():
    set_microbe_quantization_on()
    print('Microbe quantization {}'.format(get_microbe_quantization()))

def __default_case_parameters():
    return {
            'substrate_inputs' : 'DEMENTDEFAULT',
            'initial_diversity':'uniform',
            'selected_tax': None,
            'mutation_rate': None,
            "mutation_type": 'two', # one or two
            'disp_proba0':0.5,
            'boundary_flux': False,
            'diagnostic_period_day':1,
            }


def setup_ecosystem_params(eco,
                           degrad_enzyme_rng,
                           uptake_transporter_rng,
                           osmolyte_rng,
                           resp_growth_rng,
                           degrad_enzyme_kin_rng,
                           uptake_transporter_kin_rng,
                           case_parameters=__default_case_parameters()
                           ):

    print('CASE PARAMETERS')
    for k,d in case_parameters.items():
        print(k,d)

    model_functions = eco.functions_module


    # setup associations
    #######################
    # SUBSTRATRE ENZYMES
    #######################
    # Kinetic enzymatic parameters of degradation enzymes
    Ea_input = eco.degradation_enzymes.get_Ea_input_from_file(
            io.StringIO(default_data.ref_sub_enzyme_Ea))
    eco.degradation_enzymes.draw_Ea_from_bounds_df_uniform(Ea_input, degrad_enzyme_rng)
    # substrate / enzyme association

    sub_enz_d = {s:e for s,e in zip(eco.degradation_enzymes.reaction_names,
                                    eco.degradation_enzymes.enzyme_names,
                                    )}
    def sub_enz_map(sub,enz):
        return sub_enz_d[sub] == enz
    eco.degradation_enzymes.set_map_from_func(sub_enz_map)
    # Vmax and Km with linear tradeoff
    eco.degradation_enzymes.draw_VmaxKm_linear_tradeoff(
        Vmax_bounds=(25,25), # (5, 50),  # Vmax bounds
        Km_rel_error=0.0,  # Km_rel_error=0.0001, # Km_error
        Vmax_Km_slope=1.0,  # Vmax_Km_slope
        Vmax_Km_intercept=0.0,  # Vmax_Km_intercept
        Km_bounds=(0.0, None),  # Km bounds
        rng=degrad_enzyme_rng,
        specificity_tradeoff=0.0, #1.0,
    )

    # eco.degradation_enzymes.environmental_modulation['Vmax'] = (
            # model_functions.Allison_wp_factor_substrate)
    eco.degradation_enzymes.set_environmental_modulation(
        'Vmax',
        model_functions.Allison_wp_factor_substrate
        )
    ###########################
    # SUBSTRATE / MONOMERS
    ###########################
    # setup substrate monomer association
    # 1 substrate/1 organic monomer
    # NB : the following function relies on the naming scheme for monomers
    def sub_mon_map(sub, mon):
        res = (mon == sub)
        return res
    eco.substrate_degradation_op.set_map_from_func(sub_mon_map)
    # LCI slope for lignin cellulose correction factor
    eco.substrate_degradation_op.set_LCI_slope(-0.8)

    # Kinetic enzymatic parameters of uptake transporters
#    upt_Ea_bounds = [[35.0, 36.0]]
    upt_Ea_bounds = [[35.5, 35.5]]
    eco.uptake_transporters.draw_Ea_from_bounds_df_uniform(
        pd.DataFrame(data=np.array(upt_Ea_bounds, dtype=np.float64), columns=['Ea_min', 'Ea_max']),
        rng=uptake_transporter_kin_rng,
    )
    ##################################
    # MONOMERS / UPTAKE TRANSPORTERS
    ##################################
    # setup monomer/uptake transporter association
    # NB : the following function relies on the naming scheme for uptake transporters
    def mon_upt_map(mon, upt):
        res = (upt == 'Upt_{}'.format(mon))
        return res
    eco.monomer_uptake_op.set_map_from_func(mon_upt_map)

    # set 1 transporter per monomer
    eco.uptake_transporters.map[:, :] = False
    for i in range(eco.uptake_transporters.map.shape[0]):
        eco.uptake_transporters.map[i, i] = True
    eco.uptake_transporters.map_t = eco.uptake_transporters.map.astype(np.float64)
    #
    eco.uptake_transporters.draw_VmaxKm_linear_tradeoff(
        (10,10), #(1, 10),  # (1, 10) Vmax bounds
        0.0,  # 0.0001, # Km_error
        0.2,  # Vmax_Km_slope
        0.0,  # Vmax_Km_intercept
        (0.0, None),  # Km_min bound
        uptake_transporter_kin_rng,
    )

    eco.uptake_transporters.environmental_modulation['Vmax'] = (
            model_functions.Allison_wp_factor_uptake)


   ######################################
   # TAXA / (SUBSTRATES)
   ######################################
   # Taxa subsrate association will modulate
   # - enzyme production efficiency trough metabolic processes
   #   and enzyme/substrate association
   # - uptake transporer efficiency trough uptake/monomer and
   #   monomer substrate/association ( for organic monomers)


    tax_subs = model_functions.get_tax_subs_efficiency_matrix_dict(
                                               eco.microbes,
                                               eco.substrates,
                                               model_functions.taxon_selection_map_full
                                               )
    # print(tax_subs.keys())
    # Metabolic processes
    # degradation enzyme production
    meta_degenz = eco.microbes.get_metabolite('Degradation_Enzymes')
    degenz_stoechio = [1.0, 0.3, 0.0]
#    degenz_prod_bounds = (1.0e-5, 1.0e-4)
    enz_maint_cost = 5.0
    #
    # all taxa have all enzymescase_parameters['uptake_trait_c0']
    # n_degenz_min = meta_degenz.n_metabolites
    # n_degenz_max = meta_degenz.n_metabolites
    n_degenz_min = 0
    n_degenz_max = 0
    #
    meta_degenz.set_map_from_bounds((n_degenz_min, n_degenz_max), degrad_enzyme_rng)
    print('degenz {}'.format(meta_degenz.weight_map['Consti'].shape))
    tmp_w = np.zeros_like(meta_degenz.weight_map['Consti'])
    tmp_m = np.zeros_like(meta_degenz.map)
    degenz_weights = {2:1.0,1:0.5}
    for itax,subdesc in tax_subs.items():
        for ksub,v in subdesc.items():
            isub = eco.substrates.get_component_index(ksub)
            for ienz in range(eco.degradation_enzymes.n_enzymes):
                if (eco.degradation_enzymes.map[isub,ienz]):
                    tmp_w[itax,ienz] = degenz_weights[v]
                    tmp_m[itax,ienz] = True

    meta_degenz.set_map(tmp_m)
    # ind_wmap = np.zeros_like(meta_degenz.weight_map['Induci'])
    meta_degenz.set_weight_map('Consti',tmp_w)
    meta_degenz.set_weight_map('Induci',tmp_w)
    del(tmp_w)
    del(tmp_m)

    # same value for Constitutive enzyme production cost
    meta_degenz.set_ref_cost_v('Consti', 1.0e-4)
    # one value per taxon for Inducible enzyme production cost
    meta_degenz.set_ref_cost_v('Induci', 7.5e-5) #

    meta_degenz.set_target('Consti', 'Prod', degenz_stoechio, eco.degradation_enzymes)
    meta_degenz.set_target('Consti', 'Maint', [enz_maint_cost, 0.0, 0.0], eco.grid_shape)
    meta_degenz.set_target('Induci', 'Prod', degenz_stoechio, eco.degradation_enzymes)
    meta_degenz.set_target('Induci', 'Maint', [enz_maint_cost, 0.0, 0.0], eco.degradation_enzymes)

    meta_degenz.update_ref_costs()
    #  Uptake transporters production
    meta_upt = eco.microbes.get_metabolite('Uptake_Transporters')
    n_upt_min = 0
    n_upt_max =  0 #len(meta_upt.names)-2
    # upt_force_true = ['Upt_NH4', 'Upt_PO4']  # every taxon can metabolize abiotic monomers
    upt_force_true = meta_upt.names # we force every taxa to have every uptake transporter
    # selection is done through the weights
    meta_upt.set_map_from_bounds_with_constraints((n_upt_min, n_upt_max),
                                                  rng=uptake_transporter_rng,
                                                  forced_true=upt_force_true
                                                  )
    upt_c_bounds = (0.1,0.1) #(0.1, 0.1) #  same cost for everyone # (0.01, 0.1)#
    meta_upt.set_ref_costs_lhs({'Consti': upt_c_bounds, 'Induci': (0.0, 0.0)},
                               rng=uptake_transporter_rng
                               )
    tmp_w = np.zeros_like(meta_upt.weight_map['Consti'])
    tmp_m = np.zeros_like(meta_upt.map)
    uptake_weights = {2:1.0,1:0.5}
    for itax,subdesc in tax_subs.items():
        for ksub,v in subdesc.items():
            isub = eco.substrates.get_component_index(ksub)
            for iupt in range(meta_upt.n_metabolites):
                imons = [imon for imon in range(eco.monomers.n_components) if eco.monomer_uptake_op.map[imon,iupt]]
                for imon in imons:
                    if (eco.substrate_degradation_op.map[isub,imon]):
                        tmp_w[itax,iupt] = uptake_weights[v]
                        tmp_m[itax,iupt] = True
    # set weights of inorganic monomers to 1
    for kupt in ['Upt_NH4', 'Upt_PO4']:
        iupt = meta_upt.get_metabolite_index(kupt)
        tmp_w[:,iupt] = 1.0
        tmp_m[:,iupt] = True
    meta_upt.set_map(tmp_m)
    for k in ['Consti','Induci']:
        meta_upt.set_weight_map(k, tmp_w)
    del(tmp_w)
    del(tmp_m)

    #
    Uptake_maint_cost = 0.01
    meta_upt.set_target('Consti', 'Maint', [Uptake_maint_cost, 0.0, 0.0], eco.grid_shape)
    meta_upt.update_ref_costs()
    # osmolytes
    # nb there is only one osmolyte here
    # its sole impact is to add some metabolic losses to all taxa
    meta_osmo = eco.microbes.get_metabolite('Osmolytes')
    n_osmo_min, n_osmo_max = 1, 1
    meta_osmo.set_map_from_bounds((n_osmo_min, n_osmo_max), osmolyte_rng)
#    meta_osmo.set_ref_costs_lhs({'Consti': (1.0e-7, 1.0e-6), 'Induci': (1.0e-2, 1.0e-1)},
#                                osmolyte_rng
#                                )
    meta_osmo.set_ref_cost_v('Consti', 5.0e-7)
    meta_osmo.set_ref_cost_v('Induci', 5.0e-2)
    for k in ['Consti','Induci']:
        meta_osmo.set_weight_map(k,1.0)
    meta_osmo.set_target('Consti', 'Prod', [1.0, 0.0, 0.0], eco.grid_shape)
    meta_osmo.set_target('Consti', 'Maint', [5.0, 0.0, 0.0], eco.grid_shape)
    meta_osmo.set_target('Induci', 'Prod', [1.0, 0.0, 0.0], eco.grid_shape,
                         env_function=model_functions.osmo_psi_modulation,
                         )
    meta_osmo.set_target('Induci', 'Maint', [1.0, 0.0, 0.0], eco.grid_shape,
                         env_function=model_functions.osmo_psi_modulation,
                         )
    meta_osmo.update_ref_costs()
    # Respiration (uptake maintenance)
    meta_resp = eco.microbes.get_metabolite("Respiration_Growth")
    meta_resp.set_map_from_bounds((1, 1), resp_growth_rng)
    # nb : there is no real alea here due to constraints
    # (every taxon gets the unique metabolic process gene)
    meta_resp.set_ref_costs_lhs({'Consti': (0.0, 0.0), 'Induci': (1.0, 1.0)}, resp_growth_rng)
    for k in ['Consti','Induci']:
        meta_resp.set_weight_map(k,1.0)
    meta_resp.set_target('Induci', 'Maint', [1.0, 0.0, 0.0],
                         eco.grid_shape, priority=1,  # first served with priority
                         env_function=model_functions.respiration_ae
                         )
    meta_resp.update_ref_costs()



    # nbac = eco.microbes.n_taxa // 2 +1
    # print('nbac : {} nfunc {} ntax {}'.format(nbac, eco.microbes.n_taxa-nbac,eco.microbes.n_taxa))
    # eco.microbes.set_types('bacteria', eco.microbes.names[:nbac])
    # eco.microbes.set_types('fungi', eco.microbes.names[nbac:])
    eco.microbes.set_types('bacteria')
    # eco.microbes.set_types('fungi')

    # Mortality process setup
    eco.microbes.mortality_op.starvation_thresholds[:] = np.array([0.086, 0.012, 0.002])
    # eco.microbes.mortality_op.starvation_thresholds[:] = np.array([0.0, 0.0, 0.00])
    #
    # associate death probability function
    eco.microbes.mortality_op.set_death_proba_func(
            model_functions._mortality_proba_drought_ajusted_linear)

    eco.microbes.mutation_op.set_mutation_proba_func(model_functions.mutation_proba_symshift)
    # build tolerance bases on osmolytes genes

    meta_osmo = eco.microbes.get_metabolite('Osmolytes')
    osmo_all = np.sum(meta_osmo.ref_cost['Consti'], axis=1)
    osmo_all_min, osmo_all_max = np.min(osmo_all), np.max(osmo_all)
    if (osmo_all_max > osmo_all_min):
        osmo_all -= np.min(osmo_all)
        osmo_all /= np.max(osmo_all)
    else:
        osmo_all[()] = 1.0 # all taxa are drought insensitive
        # osmo_all[()] = 0.0
    # NB : as osmolyte parameters are the same for all taxa, so it drought sensitivity.
    # print('CAVEAT mortality')
    mort_pdict = {
                   'basal_death_proba': {'undefined': 0, 'bacteria': 0.001, 'fungi': 0.001},
                  # 'basal_death_proba': {'undefined': 0, 'bacteria': 0.00, 'fungi': 0.001},
                  'drought_rate': {'undefined': 0, 'bacteria': 10.0, 'fungi': 10.0},
                  'psi_threshold': {'undefined': -2, 'bacteria': -2, 'fungi': -2},
                  'drought_tolerance': {k: osmo_all for k in eco.microbes.get_taxa_types()},
                  }

    for mic_type, filt in eco.microbes.type_filters.items():
        for pname, pdict in mort_pdict.items():
            eco.microbes.mortality_op.set_mortality_parameter(pname,
                                                              mort_pdict[pname][mic_type],
                                                              filt
                                                              )
    eco.microbes.mortality_op.set_dead_matter_pool(eco.substrates, "DeadMic")

    eco.microbes.set_stoechiometry_quotas('bacteria',
                                          {'C': (0.825, 0.09),
                                           'N': (0.16, 0.04),
                                           'P': (0.015, 0.005)}
                                          )
    eco.microbes.set_stoechiometry_quotas('fungi',
                                          {'C': (0.9, 0.09),
                                           'N': (0.09, 0.04),
                                           'P': (0.01, 0.005)}
                                          )
    # connect microbial mass recycler to external pools
    eco.microbes.stoechio_balance_recycler.set_transfer_pool('N', eco.monomers, 'NH4')
    eco.microbes.stoechio_balance_recycler.set_transfer_pool('P', eco.monomers, 'PO4')
    eco.microbes.stoechio_balance_recycler.set_transfer_pool(
            'C',
            eco.microbes.metabolites['Respiration_Growth'].targets['Induci']['Maint'],
            'RespGrowth'
            )

    # linear decay operators
    # monomer leaching
    eco.set_linear_decay_operator('monomer leaching',
                                  eco.monomers,
                                  model_functions.monomer_leaching_inorg)

    # Degradation enzyme decay (loss rate)

    eco.set_linear_decay_operator('degradation enzyme decay',
                                  eco.degradation_enzymes,
                                  model_functions.degenz_decay_func,
                                  external_target=eco.substrates,
                                  target_component='DeadEnz'
                                  )

    # microbial dispersal
    max_sizes = {'undefined': 0, 'bacteria': 2, 'fungi': 50}
    # print('CAVEAT DISPERSAL')
    # max_sizes = {'undefined': 0, 'bacteria': 1000000, 'fungi': 1000000}
    for mic_type, siz in max_sizes.items():
        eco.microbes.dispersal_op.set_saturation_threshold(mic_type, 'C', siz)

    qfilt = eco.microbes.is_quantized
    disp0 = case_parameters['disp_proba0']
    qdisp_type = 'quantized'
    if "boundary_flux" in case_parameters.keys():
        if (case_parameters['boundary_flux']):
            qdisp_type = 'quantized_bdflux'
    if (qfilt):
        for mic_type in ['bacteria','fungi']:
            eco.microbes.dispersal_op.set_parameters(mic_type,
                                             qdisp_type, 0.0,
                                             disp_dist_ranges={'x': (-1, 1), 'y': (-1, 1)},
                                             disp_forced_probas= {'x':{0:disp0},'y':{0:disp0}}
                                             )
    else:
        eco.microbes.dispersal_op.set_parameters('bacteria',
                                                 'local_ratio', 0.5,
                                                 disp_dist_ranges={'x': (-1, 1), 'y': (-1, 1)},
                                                 )
        eco.microbes.dispersal_op.set_parameters('fungi',
                                                 'non_local_sharing', 0.0,
                                                 disp_dist_ranges={'x': (1, 1), 'y': (-1, 1)},
                                                 disp_forced_probas={'y': {0: 0.95}}
                                                 )


    eco.update_field_locator()

    if (case_parameters['mutation_rate'] is None):
        eco.microbes.mutation_op.deactivate()
    else:
        eco.microbes.mutation_op.activate()
        eco.microbes.mutation_op.set_mutation_proba_func(model_functions.mutation_proba_symshift)
        eco.microbes.mutation_op.mutation_prob_func_params = {'mu':case_parameters['mutation_rate']}
        if ('mutation_type' in case_parameters.keys()):
            assert(case_parameters['mutation_type'] in ['one','two'])
            eco.microbes.mutation_op.mutation_n_daughters = case_parameters['mutation_type']


    substrate_input_params = case_parameters['substrate_inputs']
    # src_funcs = {
        # 'Yearly' : model_functions.yearly_substrate_input,
        # 'Daily' : model_functions.daily_substrate_input,
        # }
    default_src_type = 'Yearly'
    if isinstance(substrate_input_params,dict):
        if ('type' in substrate_input_params.keys()):
            src_type = substrate_input_params['type']
        else:
            src_type = default_src_type
        if 'name' in substrate_input_params.keys():
            src_name = '{}SubstrateInput_{}'.format(src_type,substrate_input_params['name'])
        else:
            src_type = default_src_type
            src_name = '{}SubstrateInput'.format(src_type)
    eco.set_external_source(src_name,
                            eco.substrates,
                            model_functions.substrate_input,
                            )
    eco.source_operators[src_name].custom_parameters = substrate_input_params

    return eco






def setup_initial_conditions(eco,
                             substrate_init_rng,
                             degrad_enzyme_init_rng,
                             monomer_init_rng,
                             microbes_init_rng,
                             case_parameters=__default_case_parameters
                             ):
    # Fields initialization
    # subtrates fields init
#    sub_df_ini = eco.substrates.get_biomass_df_from_file(
#            io.StringIO(default_data.ref_initial_substrates))
    # eco.substrates.set_random_masses_from_bounds((0.0,1.0),substrate_init_rng)
#    eco.substrates.set_uniform_biomass_fields_from_df(sub_df_ini)
    # monomer field init
    eco.monomers.set_random_masses_from_bounds((0.0, 0.0), monomer_init_rng)
    # Degradation Enzymes field
    deg_enz_stoechio = eco.microbes.metabolites['Degradation_Enzymes'].rel_cost['Consti']['Prod']
    # set degradation enzymes initial biomass with stoechiometry consistent with production
    deg_enz_bounds_ini =  (0.0, 0.0)
    eco.degradation_enzymes.set_random_masses_from_bounds(deg_enz_bounds_ini,
                                                          degrad_enzyme_init_rng,
                                                          stoechio=deg_enz_stoechio)
    # microbial entities
    # draw mass everywhere
    # eco.microbes.set_random_masses_from_bounds((0.0,2.0),microbes_init_rng)
    eco.microbes.set_random_masses_from_quotas((1.0, 1.0), microbes_init_rng,ref_scale='C')
    # select taxa
    # p_tax = 1.0 / eco.microbes.n_taxa
#    p_tax = 10.0 * 1.0 / eco.microbes.n_taxa
    # p_tax = 1.0  # select everyone
    #
    # tax_f = eco.microbes.get_random_taxon_selection(p_tax, microbes_init_rng)
    # tax_f = tax_f.astype(eco.microbes.mass.dtype)
    # print('AAAAAAAAAAAAAAA')
    # print(tax_f.shape)
    tax_f = np.ones_like(eco.microbes.tmass[:,:,:,0])
    if (case_parameters['initial_diversity'] != 'uniform'):
        if (case_parameters['initial_diversity'] == 'dirac'): # for backward compatibility
            tax_ratios = np.zeros((eco.microbes.n_taxa,))
            if (case_parameters['selected_tax'] is not None):
                tax_ratios[case_parameters['selected_tax']] = 1.0
        elif (isinstance(case_parameters['initial_diversity'],list)):
            tax_ratios = np.array(case_parameters['initial_diversity'])
        elif(isinstance(case_parameters['initial_diversity'], dict)):
            tax_ratios = np.zeros((eco.microbes.n_taxa,))
            for k,d in case_parameters['initial_diversity'].items():
                tax_ratios[int(k)] = d
        assert(tax_ratios.shape == (eco.microbes.n_taxa,))
        tax_ratios = tax_ratios / np.sum(tax_ratios)
        tf = np.sum(tax_f,axis=-1)
        tax_f[:,:,:] = tf[:,:,np.newaxis] * tax_ratios[np.newaxis,np.newaxis,:]
    #
    for ia in range(eco.microbes.mass.shape[-1]):
        eco.microbes.mass[:, :, :, ia] = eco.microbes.mass[:, :, :, ia] * tax_f
    #
    scal = np.ones_like(eco.microbes.mass[0, 0, :, 0])
    max_sizes = {'undefined': 0, 'bacteria': 2, 'fungi': 50}
    for mic_type, filt in eco.microbes.type_filters.items():
        scal[filt] = 0.5 * max_sizes[mic_type]
    #
    eco.microbes.mass = eco.microbes.mass * scal[np.newaxis, np.newaxis, :, np.newaxis]
    eco.microbes.update_biomass_fields()

    # preliminary setup of number of individuals.
    # TODO : when quantized version is stabilized, setup a more consistent
    # scheme for initiating population ( set numbers and mass excess separately and
    # build mass from that)
    if (eco.microbes.is_quantized):
        eco.microbes.quanta[:,:,:,0] = np.floor(eco.microbes.mass[:,:,:,0] / scal[np.newaxis, np.newaxis, :])

    return eco


def set_climate(eco):

    model_functions = eco.functions_module
    eco.set_environment_function(model_functions.env_func_constant)

    return eco



def setup_diagcollector(diagcollector, period_day=1):
    """
    Defines functions to collect light data diagnostics and setup timelines

    Parameters
    ----------
    diagcollector : :class:`dementmutant.ecosystem.DiagnosticCollector`
        An intialized `DiagnosticCollector` instance, as obtained
        by :meth:`dementmutant.ecosystem.Ecosystem.get_diag_collector`

    Returns
    -------
    :class:`dementmutant.ecosystem.DiagnosticCollector`
        A  `DiagnosticCollector` instance with a collection
        of various aggregated data timelines


    """
    def space_sum(f, itime):
        if  (not(itime%period_day == 0)):
            return None
        res = np.sum(f.mass, axis=f.get_space_axes())
        return res

    def space_component_sum(f, itime):
        if  (not(itime%period_day == 0)):
            return None
        res = np.sum(f.mass, axis=(0, 1, 2))
        return res

    def get_full_sum_by_type(f, itime):
        if  (not(itime%period_day == 0)):
            return None
        sums = []
        for mic_type, filt in f.type_filters.items():
            s = np.sum(f.mass[:, :, filt, :], axis=(0, 1, 2))
            sums.append(s)
        res = np.row_stack(tuple(sums))
        return res

    def get_ncells_per_comp(f, itime):
        if  (not(itime%period_day == 0)):
            return None
        spax = f.get_space_axes()
        res = np.squeeze(np.sum((f.tmass > 0).astype(int), axis=spax))
        return res

    def get_nnz_comp_per_grid_cell(f, itime):
        if  (not(itime%period_day == 0)):
            return None
        r1 = np.sum(np.squeeze((f.tmass > 0).astype(int)), axis=-1).astype(np.float_)
        m = np.mean(r1)
        sig = np.std(r1)
        m1, m2 = np.min(r1), np.max(r1)
        res = np.array([m, sig, m1, m2])
        return res

    def climate(eco, itime):
        if  (not(itime%period_day == 0)):
            return None
        env = eco.get_environment(itime)
        return env

    def quanta_space_sum(f, itime):
        if  (not(itime%period_day == 0)):
            return None
        if (f.is_quantized):
            res = np.sum(f.quanta, axis=f.get_space_axes())
            return res
        else:
            return None
    def mean_taxon_pos_space_stats(f,itime):
        if  (not(itime%period_day == 0)):
            return None
        if (f.is_quantized):
            tax_ids = np.array(list(range(f.quanta.shape[2])))
            d1 = np.sum((f.quanta[:,:,:,0] * tax_ids[np.newaxis,np.newaxis,:]),axis=-1)
            # d1 = d1 / np.sum(f.quanta[:,:,:,0],axis=-1)
            den = np.sum(f.quanta[:,:,:,0],axis=-1)
            filt = den > 0
            if (np.any(filt)):
                d1 = d1[filt]
                den = den[filt]
                d1 = d1 /den
                # d1 = np.divide(d1,den,where=den > 0)
                res = np.array([np.mean(d1),np.std(d1),np.min(d1),np.max(d1)])
            else:
                res = np.array([np.NaN,np.NaN,np.NaN,np.NaN])
            return res

    def mean_taxon_pos_massweighted_space_stats(f,itime):
        if  (not(itime%period_day == 0)):
            return None
        tm = np.sum(f.mass,axis=-1)
        tax_ids = np.array(list(range(tm.shape[-1])))
        den = np.sum(tm,axis=-1)
        d1 = np.sum(tm * tax_ids[np.newaxis,np.newaxis,:],axis=-1)
        filt = den > 0
        if (np.any(filt)):
            d1 = d1[filt]
            den = den[filt]
            d1 = d1 / den
            res = np.array([np.mean(d1),np.std(d1),np.min(d1),np.max(d1)])
        else:
            res = np.array([np.NaN,np.NaN,np.NaN,np.NaN])
        return res

    def taxon_individual_mass_stats(f,itime):
        if  (not(itime%period_day == 0)):
            return None
        tm = np.sum(f.mass,axis=-1)
        quants = f.quanta[:,:,:,0]
        res = np.zeros((quants.shape[-1],4))
        for itax in range(quants.shape[-1]):
            filt = quants[:,:,itax] > 0
            if np.any(filt):
                d1 = tm[:,:,itax][filt] / quants[:,:,itax][filt]
                res[itax,:] = np.array([np.mean(d1),np.std(d1),np.min(d1),np.max(d1)])
        return res

    diagcollector.set_field_timeline_desc('space_sum', space_sum)
    diagcollector.set_field_timeline_desc('space_comp_sum', space_component_sum)
    diagcollector.set_field_timeline_desc('sum_by_type', get_full_sum_by_type, ['microbes', ])
    diagcollector.set_field_timeline_desc('ncells', get_ncells_per_comp, ['microbes', ])
    diagcollector.set_field_timeline_desc('ntaxpercell', get_nnz_comp_per_grid_cell, ['microbes', ])
    diagcollector.set_global_timeline_desc('Climate', climate)

    diagcollector.set_field_timeline_desc('Quanta', quanta_space_sum,['microbes',])
    diagcollector.set_field_timeline_desc('LocalMeanTaxPos',
                                          mean_taxon_pos_space_stats,
                                          ['microbes',]
                                          )
    diagcollector.set_field_timeline_desc('LocalMeanTaxPosMassWeighted',
                                          mean_taxon_pos_massweighted_space_stats,
                                          ['microbes',]
                                          )
    diagcollector.set_field_timeline_desc('LocalTaxIndividualMass',
                                          taxon_individual_mass_stats,
                                          ['microbes',]
                                          )

    def boundary_flux_stats(eco,itime):
        res = {}
        for mic_type,d in eco.microbes.dispersal_op.last_run_stats.items():
            for k,d in d.items():
                kk = '{}_{}'.format(mic_type,k)
                res[kk] = np.array(d)
        # print(res)
        return res

    diagcollector.set_global_timeline_desc('BoundaryFlux',boundary_flux_stats)


    def birth_mut_stats(eco,itime):
        res = {k:np.array(d)
               for k, d in eco.microbes.mutation_op.last_run_stats.items()
               }
        return res

    diagcollector.set_global_timeline_desc('BirthMutation', birth_mut_stats)

    def TaxClassPopulations(eco,itime):
        f = eco.functions_module.taxon_selection_map
        tax_classes = {k : np.array([i for i in range(eco.microbes.n_taxa) if f(i)[0]== k])
                       for k in ['MICr','MICk']}
        tmp_sum = np.sum(eco.microbes.quanta,axis=(0,1))[:,0]
        res = {k:np.sum(tmp_sum[d]) for k, d in tax_classes.items()}
        return res
    diagcollector.set_global_timeline_desc('TaxClassesPop',TaxClassPopulations)

    def SubClassTMasses(eco,itime):
        submass = np.sum(eco.substrates.tmass,axis=(0,1))
        subclasses = eco.functions_module.get_substrate_classes_dict()
        subinds = {k:np.array([eco.substrates.get_component_index(sub) for sub in d]) for k,d in subclasses.items()}
        res = {k: np.sum(submass[d]) for k,d in subinds.items()}
        return res
    diagcollector.set_global_timeline_desc('SubClassesTMass',SubClassTMasses)


    def EnzClassTmasses(eco,itime):
        enzmass = np.sum(eco.degradation_enzymes.tmass,axis=(0,1))
        subclasses = eco.functions_module.get_substrate_classes_dict()
        subinds = {k:np.array([eco.substrates.get_component_index(sub) for sub in d]) for k,d in subclasses.items()}
        res = {k: np.sum(enzmass[d]) for k,d in subinds.items()}
        return res

    diagcollector.set_global_timeline_desc('EnzClassesTMass', EnzClassTmasses)

    # for debbgging purpose
    # def get_peak_mem(eco,itime):
    #     if  (not(itime%period_day == 0)):
    #         return None
    #     pid = os.getpid()
    #     pp  = psutil.Process(pid)
    #     memi = pp.memory_info()
    #     res = {k:getattr(memi, k) for k in ['rss','vms','shared','text','data','lib']}
    #     return res
    # diagcollector.set_global_timeline_desc('Memory', get_peak_mem)


    return diagcollector



def prepare_simu_from_seeds(grid_shape, n_degrad_enzymes, n_taxa, n_osmolytes,
                            param_seeds=None,
                            init_seeds=None,
                            case_parameters = __default_case_parameters()
                            ):
    """
    Setup a generic simulation from a set of random seeds
    """

    assert(n_degrad_enzymes == 12)
    eco = get_default_empty_ecosystem(grid_shape,
                                      n_degrad_enzymes,
                                      n_taxa,
                                      n_osmolytes,
                                      )
    if (param_seeds is None):
        param_seeds = {
                'degrad_enzyme': 0,
                'uptake_transporter': 0,
                'degrad_enzyme_kin': 0,
                'uptake_transporter_kin': 0,
                'osmolyte': 0,
                'resp_growth': 0,
                }
    param_rng = {k+'_rng': np.random.default_rng(d) for k, d in param_seeds.items()}

    eco = setup_ecosystem_params(eco, **param_rng,
                                 case_parameters=case_parameters,
                                 )

    if (init_seeds is None):
        init_seeds = {
                "substrate": 0,
                "degrad_enzyme": 0,
                "monomer": 0,
                "microbes": 0,
                }
    init_rng = {k+'_init_rng': np.random.default_rng(d) for k, d in init_seeds.items()}

    eco = setup_initial_conditions(eco, **init_rng,
                                   case_parameters=case_parameters)
    eco = set_climate(eco)

    eco.update_field_locator()
    return eco


def build_simu_replica(grid_shape,
               n_degrad_enzymes,
               n_taxa,
               n_osmolytes,
               param_seeds=None,
               init_seeds=None,
               run_seeds=None,
               root_dir='.',
               case_parameters = __default_case_parameters()
               ):
    eco = prepare_simu_from_seeds(grid_shape,
                                  n_degrad_enzymes,
                                  n_taxa,
                                  n_osmolytes,
                                  param_seeds,
                                  init_seeds,
                                  case_parameters = case_parameters,
                                  )

    if run_seeds is None:
        run_seeds = {
                'microbes_mortality': 0,
                'microbes_dispersal': 0,
                'microbes_mutation': 0,
                }
    def subdir_name_from_params(cp):
        if (cp['mutation_rate'] is None):
            mut_tag = 'Nomut'
        else:
            mut_tag = 'Mut'
        init_div_tag = cp['initial_diversity']
        if (cp['initial_diversity'] == 'dirac'):
            init_div_tag += str(cp['selected_tax'])
        bdresampling_tag = {False:'NoBFR',True:'BFR'}[cp['boundary_flux']]
        sd = 'input_{}_{}_{}_{}_{}'.format(cp['substrate_inputs']['name'],
                                        cp['substrate_inputs']['type'],
                                        mut_tag,
                                        init_div_tag,
                                        bdresampling_tag,
                                        )
        return sd
    substrate_dir = subdir_name_from_params(case_parameters)
    sub_dir = 'replica_{}_{}_{}'.format(*tuple(run_seeds.values()))
    rep_dir = os.path.join(root_dir,substrate_dir,sub_dir)
    os.makedirs(rep_dir,exist_ok=True)
    set_run_seed_in_directory(run_seeds, rep_dir)
    create_simulation_directory(eco, rep_dir)



def build_and_run_simu(grid_shape,
                       n_degrad_enzymes,
                       n_taxa,
                       n_osmolytes,
                       n_steps,
                       param_seeds=None,
                       init_seeds=None,
                       run_seeds=None,
                       with_diags=False,
                       save_fields=False,
                       fields_saving_period=1,
                       save_dir='.',
                       num_threads=1,
                       save_diags=False,
                       case_parameters = __default_case_parameters()
                       ):

    eco = prepare_simu_from_seeds(grid_shape,
                                  n_degrad_enzymes,
                                  n_taxa,
                                  n_osmolytes,
                                  param_seeds,
                                  init_seeds,
                                  case_parameters=case_parameters
                                  )

    if run_seeds is None:
        run_seeds = {
                'microbes_mortality': 0,
                'microbes_dispersal': 0,
                'microbes_mutation': 0,
                }

    if (save_fields or save_diags):
        set_run_seed_in_directory(run_seeds, save_dir)

    run_rng = {k: np.random.default_rng(d) for k, d in run_seeds.items()}
    print('Using run seeds {}'.format(run_seeds))
    if (with_diags):
        diagcollector = eco.get_diag_collector()
        diagcollector = setup_diagcollector(diagcollector,
                                            period_day=case_parameters['diagnostic_period_day'])
    else:
        diagcollector = None

    if (save_fields or save_diags):
        create_simulation_directory(eco, save_dir)

    eco, diagcollector = run_simulation_time_loop(eco,
                                                  n_steps, run_rng,
                                                  diagcollector=diagcollector,
                                                  save_fields=save_fields,
                                                  fields_saving_period=fields_saving_period,
                                                  save_dir=save_dir,
                                                  num_threads=num_threads,
                                                  )
    if (with_diags) and (save_diags):
        dump_diagcollector(diagcollector, save_dir, n_steps)

    return eco, diagcollector


def get_sim_files(save_dir):
    res = get_simulation_files(save_dir)
    return res


def reload_simulation(save_dir, itime=0, eco=None,
                      functions_module_name='model_functions',
                      ):
    """
    Reloads simulation from data saved in `save_dir`
    in its initial state (time index 0)

    Parameters
    ----------
        save_dir : str
            Path of the directory where the simulation was saved

    """
    if (eco is None):
        mod_path = os.path.join(save_dir,"model_functions.py")
        eco = load_simu_from_dir(save_dir, itime,
                                 functions_module_name=functions_module_name,
                                 functions_module_path=mod_path,
                                 )
    else:
        eco = reload_fields_state(eco, save_dir, itime)
    return eco
