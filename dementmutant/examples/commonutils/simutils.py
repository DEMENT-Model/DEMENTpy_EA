#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A set of common functions to create and setup simulations
used in the various workflow examples.

Created on Friday 28 15:08:05 2022

Copyright CNRS 2021

@author: david.coulette@ens-lyon.fr

"""

import sys
import os
import psutil
import resource
import glob
import numpy as np
from numba import set_num_threads, get_num_threads

sys.path.append('../../')
from dementmutant.defaults import _substrates_default, _inorganic_monomers_default
from dementmutant.ecosystem import Ecosystem, ecosystem_from_json_file
from dementmutant.utility import StageTimer, dict_from_json, dict_to_json

#: str Naming template for json run seed files
json_runseeds_template_str = '{}/run_seeds.json'
#: str Naming template for json ecosystem parameter files
json_param_template_str = '{}/ecosystem_parameters.json'
#: str Naming template for hdf5 biomass fields files
hdf5_fields_dump_template_str = '{}/biomassfields_dump_{:06d}.hdf5'
#: str Naming template for hdf5 timelines files
hdf5_timelines_dump_template_str = '{}/timelines_dump_{:06d}_{:06d}.hdf5'


# TODO ? add formal descriptor class for time scheme steps

__stoechiometry_step_exec_types = {'standalone','with_mortality'}

__stoechiometry_step_exec = 'with_mortality'


def set_stoechimetry_step_type(step_type):
    if (step_type in __stoechiometry_step_exec_types):
        global __stoechiometry_step_exec
        __stoechiometry_step_exec = step_type
    else:
        print('Unkown stoechiometry step type - valid ones are {}'.format(
            __stoechiometry_step_exec_types))
def get_stoechimetry_step_type():
    global __stoechiometry_step_exec
    return  __stoechiometry_step_exec


def _flushout():
    if (not hasattr(sys, 'ps1')):
        sys.stdout.flush()


def get_default_empty_ecosystem(grid_shape,
                                n_degrad_enzymes,
                                n_taxa,
                                n_osmolytes,
                                subs_names=None,
                                ):
    """
    Setup a generic empty simulation with standardized names
    for the various entities

    Substrates are the 12 default ones used in the DEMENT model
    Monomers are the 14 default ones used in the DEMENT model
    (organic ones have the same name as substrates and the two
    inorganic ones are NH4 and PO4)

    The number of uptake transporters is equal to the number of
    monomer (with a 1-1 association)

    Parameters
    ----------

    grid_shape : 2-tuple of int
        dimensions of spatial grid

    n_degrad_enzymes : int
        number of degradation enzyme entities

    n_taxa : int
        number of microbial taxa

    n_osmolytes : int
        number of osmolyte entities

    subs_names : list of strings (optional)
        list of substrates names (defaults to the default dement model subsrates)
        CAVEAT : Some mandatory substrates have to be present
                - DeadMic for mortality processes (recycling)
                - DeadEnz for enyme decay ( enzymes recycling)
                - Cellulose/Lignin for the correction factor in substrate degradation

    Returns
    -------

    :class:`dementmutant.ecosystem.Ecosystem`
        An empty `Ecosystem` object with all fields set
        to zero and unitiliazed parameters

    Warning
    -------
        In that state the system is unfit to run

    """
    sub_names = _substrates_default
    mon_names = _inorganic_monomers_default + sub_names
    degenz_names = ['DegEnz{}'.format(i) for i in range(1, n_degrad_enzymes+1)]
    upt_names = ['Upt_{}'.format(s) for s in mon_names]
    osm_names = ['Osmo_{}'.format(i) for i in range(1, n_osmolytes+1)]
    taxa_names = ['Tax_{}'.format(i) for i in range(n_taxa)]

    eco = Ecosystem(
                    grid_shape=grid_shape,
                    substrate_names=sub_names,
                    degradation_enzyme_names=degenz_names,
                    monomer_names=mon_names,
                    uptake_transporter_names=upt_names,
                    osmolyte_names=osm_names,
                    taxa_names=taxa_names,
                    )
    return eco


def run_simulation_time_loop(eco, n_steps, run_rng,
                             diagcollector=None,
                             save_fields=False,
                             fields_saving_period=1,
                             save_dir='.',
                             display_runtimes=True,
                             display_steps_messages=True,
                             display_step_period=100,
                             num_threads=1,
                             ):
    """
    Timeloop - Run `n_steps` days


    Parameters
    ----------

    eco : :class:`dementmutant.ecosystem.Ecosystem`
        A fully initialized  `Ecosystem` instance with parameters
        and initial conditions fully prepared.

    n_steps : int
        Number of steps (days) to run

    run_rng : dict of :class:`numpy.random.Generator`
        Properly seeded Random generators objects for this run stored in a per
        process dictionnary.
        The keys must be  ['microbes_mortality','microbes_dispersal','microbes_mutation']

        Note
        ----
        All values can point to the same generator if you don't need fine
        control on stochasticity of individual processes
        (as for fine grained variance analysis)

    diagcollector : :class:`dementmutant.ecosystem.DiagnosticCollector` , optional
        An optional already setup`DiagnosticCollector` object

    save_fields : bool
        Switch saving of biomassfields to hdf5.

    fields_saving_period : int
        Period in time steps for dump of biomassfields to hdf5

    save_dir : str
        Path of the directory to save biomassfields

    display_runtimes : bool
        Switch on/off the displaying of runtimes measurements
        Measurement are taken for the first 2 time steps and the last one
        The first step is usually longer as numba kernels are jit-compiled
        the first time they are called.

    num_threads :  int

        Number of threads to use for parallel numba kernels
        If set to 'Auto' , will use all avaiable threads (up to
        the number of avaiable cores or the value of the NUMBA_NUM_THREADS
        environment variable if it is set)
        Note that for small cases, computations are memory-bounds, so you won't
        get aby speed-up by running on multiple-threads.
        Perform a few tests with a few time steps and values of `num_threads` from
        1 to the number of cores on your machine to find the optimal value.
        Note that this value is problem size dependent.

    Returns
    -------

    :class:`dementmutant.ecosystem.Ecosystem`
        The input `eco`  with state obtained after n_steps

    :class:`dementmutant.ecosystem.DiagnosticCollector` | None
        The input `diagcollector` with recorder data timelines.
        Note that `diagcollector` timelines are not saved to disk
        by this routine. It is up to the caller to do it if required.

    """

    save_last_fields = True # TODO  ? add as argument ?

    itime = 0

    if (diagcollector is not None):
        diagcollector.init_timelines()
        diagcollector.record_timelines(itime)

    if (save_fields):
        os.makedirs(save_dir, exist_ok=True)
        eco.field_locator.add_tags_to_fields(['save', ])
        hdf_file_path = hdf5_fields_dump_template_str.format(save_dir, itime)
        eco.save_biomass_fields_to_hdf5(hdf_file_path)
        json_param_path = '{}/ecosystem_parameters.json'.format(save_dir)
        eco.save_parameters_to_json(json_param_path)
        eco.save_functions_module(save_dir)

    st_glob = StageTimer()
    msg_period = 100

    print('Starting simulation')
    _flushout()
    av_n_threads = get_num_threads()
    set_num_threads(num_threads)
    prov_n_threads = get_num_threads()
    msg_str = 'Number of threads Required {} Available {} Provided {}'
    print(msg_str.format(num_threads, av_n_threads, prov_n_threads))
    _flushout()
    for itime in range(n_steps):
        if ((itime % msg_period) == 0) and (itime > 0):
            print('Step {}'.format(itime))
            _flushout()
            st_glob.tag_event('Steps {}-{}'.format(itime-msg_period, itime))
        st = StageTimer()
        eco.apply_monomer_diffusion(itime)
        st.tag_event('monomer diffusion')
        eco.apply_subtrate_degradation(itime)
        st.tag_event('substrate degradation')
        eco.apply_monomer_uptake_only(itime)
        st.tag_event('uptake only')
        eco.apply_inducible_only(itime)
        st.tag_event('metabolic inducible')
        eco.apply_metabolic_processes('Consti', itime)
        st.tag_event('metabolic constitutive')
        #
        st_type = get_stoechimetry_step_type()
        if (st_type == 'standalone'):
            eco.apply_stoechiometry_only(itime)
            st.tag_event("Standalone Stoechiometry")
        #
        if (eco.microbes.is_quantized):
            eco.apply_quantized_birth_mut(itime, run_rng['microbes_mutation'])
            st.tag_event('Quantized birth/mutation')
        #
        if (st_type == 'standalone'):
            eco.apply_mortality_only(itime, run_rng['microbes_mortality'])
            st.tag_event('mortality only')
        if (st_type == 'with_mortality'):
            eco.apply_mortality_stoech(itime, run_rng['microbes_mortality'])
            st.tag_event('mortality + stoechiomoetry')
        eco.apply_linear_decay_ops(itime)
        st.tag_event('Linear decay (monomers, enzymes)')
        eco.apply_dispersal(itime,
                            run_rng['microbes_dispersal'],
                            run_rng['microbes_mutation'])
        st.tag_event("Microbial dispersal")
        # eco.apply_dispersal_phase1(itime,
        #                             run_rng['microbes_dispersal'],
        #                             run_rng['microbes_mutation'])
        # st.tag_event('dispersal phase 1')
        # eco.apply_dispersal_phase2(itime,
        #                             run_rng['microbes_dispersal'],
        #                             run_rng['microbes_mutation'])
        # st.tag_event('dispersal phase 2')

        eco.apply_external_sources(itime)
        st.tag_event('external sources')
        if (diagcollector is not None):
            diagcollector.record_timelines(itime+1)
            st.tag_event('diagnostics')
        sf_switch = save_fields and (0 == ((itime+1) % fields_saving_period))
        sf_switch = sf_switch or ((itime == n_steps-1) and save_last_fields)
        if (sf_switch):
            hdf_file_path = hdf5_fields_dump_template_str.format(save_dir, itime+1)
            eco.save_biomass_fields_to_hdf5(hdf_file_path)
            st.tag_event('hdf5 file saving')

        st.set_end()

        if (itime in [0, 1, n_steps-1]):
            if (display_runtimes):
                print('Detailed Timings for step {}'.format(itime))
                st.display()

        st_glob.tag_event('Step {}'.format(itime))

    st_glob.set_end()
    if (display_runtimes):
        print(' Global timings')
        st_glob.display(show_stages=False)
        _flushout()

    return eco, diagcollector

# TODO DEPRECATE DIRECT usage
# REPLACE by more generic example, to be copied and modfied
# def setup_diagcollector_example(diagcollector, period_day=1):
#     """
#     Defines functions to collect light data diagnostics and setup timelines

#     Parameters
#     ----------
#     diagcollector : :class:`dementmutant.ecosystem.DiagnosticCollector`
#         An intialized `DiagnosticCollector` instance, as obtained
#         by :meth:`dementmutant.ecosystem.Ecosystem.get_diag_collector`

#     Returns
#     -------
#     :class:`dementmutant.ecosystem.DiagnosticCollector`
#         A  `DiagnosticCollector` instance with a collection
#         of various aggregated data timelines


#     """
#     def space_sum(f, itime):
#         if  (not(itime%period_day == 0)):
#             return None
#         res = np.sum(f.mass, axis=f.get_space_axes())
#         return res

#     def space_component_sum(f, itime):
#         if  (not(itime%period_day == 0)):
#             return None
#         res = np.sum(f.mass, axis=(0, 1, 2))
#         return res

#     def get_full_sum_by_type(f, itime):
#         if  (not(itime%period_day == 0)):
#             return None
#         sums = []
#         for mic_type, filt in f.type_filters.items():
#             s = np.sum(f.mass[:, :, filt, :], axis=(0, 1, 2))
#             sums.append(s)
#         res = np.row_stack(tuple(sums))
#         return res

#     def get_ncells_per_comp(f, itime):
#         if  (not(itime%period_day == 0)):
#             return None
#         spax = f.get_space_axes()
#         res = np.squeeze(np.sum((f.tmass > 0).astype(int), axis=spax))
#         return res

#     def get_nnz_comp_per_grid_cell(f, itime):
#         if  (not(itime%period_day == 0)):
#             return None
#         r1 = np.sum(np.squeeze((f.tmass > 0).astype(int)), axis=-1).astype(np.float)
#         m = np.mean(r1)
#         sig = np.std(r1)
#         m1, m2 = np.min(r1), np.max(r1)
#         res = np.array([m, sig, m1, m2])
#         return res

#     def climate(eco, itime):
#         if  (not(itime%period_day == 0)):
#             return None
#         env = eco.get_environment(itime)
#         return env

#     def quanta_space_sum(f, itime):
#         if  (not(itime%period_day == 0)):
#             return None
#         if (f.is_quantized):
#             res = np.sum(f.quanta, axis=f.get_space_axes())
#             return res
#         else:
#             return None
#     def mean_taxon_pos_space_stats(f,itime):
#         if  (not(itime%period_day == 0)):
#             return None
#         if (f.is_quantized):
#             tax_ids = np.array(list(range(f.quanta.shape[2])))
#             d1 = np.sum((f.quanta[:,:,:,0] * tax_ids[np.newaxis,np.newaxis,:]),axis=-1)
#             # d1 = d1 / np.sum(f.quanta[:,:,:,0],axis=-1)
#             den = np.sum(f.quanta[:,:,:,0],axis=-1)
#             filt = den > 0
#             d1 = d1[filt]
#             den = den[filt]
#             d1 = d1 /den
#             # d1 = np.divide(d1,den,where=den > 0)
#             res = np.array([np.mean(d1),np.std(d1),np.min(d1),np.max(d1)])
#             return res

#     def mean_taxon_pos_massweighted_space_stats(f,itime):
#         if  (not(itime%period_day == 0)):
#             return None
#         tm = np.sum(f.mass,axis=-1)
#         tax_ids = np.array(list(range(tm.shape[-1])))
#         den = np.sum(tm,axis=-1)
#         d1 = np.sum(tm * tax_ids[np.newaxis,np.newaxis,:],axis=-1)
#         filt = den > 0
#         d1 = d1[filt]
#         den = den[filt]
#         d1 = d1 / den
#         res = np.array([np.mean(d1),np.std(d1),np.min(d1),np.max(d1)])
#         return res

#     def taxon_individual_mass_stats(f,itime):
#         if  (not(itime%period_day == 0)):
#             return None
#         tm = np.sum(f.mass,axis=-1)
#         quants = f.quanta[:,:,:,0]
#         res = np.zeros((quants.shape[-1],4))
#         for itax in range(quants.shape[-1]):
#             filt = quants[:,:,itax] > 0
#             if not(np.any(filt)):
#                 continue
#             d1 = tm[:,:,itax][filt] / quants[:,:,itax][filt]
#             res[itax,:] = np.array([np.mean(d1),np.std(d1),np.min(d1),np.max(d1)])
#         return res

#     diagcollector.set_field_timeline_desc('space_sum', space_sum)
#     diagcollector.set_field_timeline_desc('space_comp_sum', space_component_sum)
#     diagcollector.set_field_timeline_desc('sum_by_type', get_full_sum_by_type, ['microbes', ])
#     diagcollector.set_field_timeline_desc('ncells', get_ncells_per_comp, ['microbes', ])
#     diagcollector.set_field_timeline_desc('ntaxpercell', get_nnz_comp_per_grid_cell, ['microbes', ])
#     diagcollector.set_global_timeline_desc('Climate', climate)

#     diagcollector.set_field_timeline_desc('Quanta', quanta_space_sum,['microbes',])
#     diagcollector.set_field_timeline_desc('LocalMeanTaxPos',
#                                           mean_taxon_pos_space_stats,
#                                           ['microbes',]
#                                           )
#     diagcollector.set_field_timeline_desc('LocalMeanTaxPosMassWeighted',
#                                           mean_taxon_pos_massweighted_space_stats,
#                                           ['microbes',]
#                                           )
#     diagcollector.set_field_timeline_desc('LocalTaxIndividualMass',
#                                           taxon_individual_mass_stats,
#                                           ['microbes',]
#                                           )

#     # specific to cellulase production/uptake
#     def mean_cellulase_traits(eco,itime):
#         if  (not(itime%period_day == 0)):
#             return None
#         metnames = ['Degradation_Enzymes','Uptake_Transporters']
#         proctypes = ['Induci','Consti']
#         cellulose_select = ['DegEnz3','Upt_Cellulose']
#         funcs_d = {'mean':np.mean,'std':np.std,'min':np.min,'max':np.max}
#         qq = eco.microbes.quanta[:,:,:,0]
#         den = np.sum(qq,axis=-1)
#         filt = den > 0
#         den = den[filt]
#         res = {}
#         for metname,proc,select in zip(metnames,proctypes,cellulose_select):
#             met = eco.microbes.metabolites[metname]
#             i_cellulose = met.get_metabolite_index(select)
#             w = met.ref_cost[proc][:,i_cellulose]
#             d1 = np.sum(qq * w[np.newaxis,np.newaxis,:],axis=-1)
#             d1 = d1[filt]
#             d1 = d1 / den
#             # d1 = np.divide(d1,den,where=den > 0)
#             for kf,f in funcs_d.items():
#                 k = '{}_{}_{}'.format(proc,select,kf)
#                 res[k] = f(d1)
#         return res

#     diagcollector.set_global_timeline_desc('CelluloseTraits',
#                                             mean_cellulase_traits,
#                                             )

#     # for debbgging purpose
#     def get_peak_mem(eco,itime):
#         if  (not(itime%period_day == 0)):
#             return None
#         pid = os.getpid()
#         pp  = psutil.Process(pid)
#         memi = pp.memory_info()
#         res = {k:getattr(memi, k) for k in ['rss','vms','shared','text','data','lib']}
#         #ress = resource.getrusage(resource.RUSAGE_SELF)
#         #res = {k:getattr(ress, k) for k in ['ru_maxrss','ru_ixrss','ru_idrss']}
#         return res
#     diagcollector.set_global_timeline_desc('Memory', get_peak_mem)
#     return diagcollector


def dump_diagcollector(diagcollector, save_dir, n_steps):
    """
        Save Diagcollector recorded data in hdf5 file

    Parameters
    ----------
        diagcollector : :class:`dementmutant.ecosystem.DiagnosticCollector`
            DiagnosticCollector object
        save_dir : str
            Path of the file to save data to
        n_steps : int
            Number of steps of the simulation.
            :note: This parameter is only used for naming the file
            using the default file template

    Returns
    -------
        bool
            True if all went well

    """

    os.makedirs(save_dir, exist_ok=True)
    diagcollector.dump_all_timelines_to_hdf5(
                hdf5_timelines_dump_template_str.format(save_dir, 0, n_steps)
                                                )
    return True


def run_simu_from_dir(n_steps,
                      save_dir='.',
                      save_fields=False,
                      fields_saving_period=1,
                      setup_diagcollector=None,
                      display_runtimes=True,
                      display_steps_messages=True,
                      display_step_period=100,
                      num_threads=1,
                      functions_module_name = "model_functions",
                      functions_module_path = None,
                      ):
    """
    This functions loads simulation parameters (json format) and initial
    states of biomassfields from hdf5 files.
    Its runs the simulation and perform saving of state and timelines if
    required.
    It is meant to be used for sequences of runs in batch mode in a workflow
    when simulation directories are setup in a first step, then runs are performed
    individually (either in one loop from a script, mutliprocess spawning,
    individual tasks submitted to a bactch submission system)

    Parameters
    ----------

    n_steps : int
        number of time_steps to run

    save_dir : str
        Path of the directory to load and save results

    fields_saving_period : int
        Period in time steps for dump of biomassfields to hdf5

    setupdiagcollector : func
        function that takes an initialized :class:`dementmutant.ecosystem.DiagnosticCollector`
        instance, define its timelines diagnostics routines, and returns the object.

    display_runtimes : bool
        Switch on/off the displaying of runtimes measurements
        Measurement are taken for the first 2 time steps and the last one
        The first step is usually longer as numba kernels are jit-compiled
        the first time they are called.

    num_threads :  int
        Number of threads to use for parallel numba kernels
        If set to 'Auto' , will use all avaiable threads (up to
        the number of avaiable cores or the value of the NUMBA_NUM_THREADS
        environment variable if it is set)
        Note that for small cases, computations are memory-bounds, so you won't
        get aby speed-up by running on multiple-threads.
        Perform a few tests with a few time steps and values of `num_threads` from
        1 to the number of cores on your machine to find the optimal value.
        Note that this value is problem size dependent.


    Returns
    -------
        bool
            True if everything went well

    """
    json_param_path = json_param_template_str.format(save_dir)
    if (functions_module_path is None):
        print('SIMUTILS WARNING FUNCTION MODULE PATH ABSENT AAAAAA')
        sys.path.append(save_dir)
    eco = ecosystem_from_json_file(json_param_path,
                                   functions_module_name=functions_module_name,
                                   functions_module_path=functions_module_path,
                                   )

    json_runseeds_path = json_runseeds_template_str.format(save_dir)
    if os.path.isfile(json_runseeds_path):
        run_seeds = dict_from_json(json_runseeds_path)
        print('Loading seeds')
        print(run_seeds)
    else:
        run_seeds = {
                      'microbes_mortality': 0,
                      'microbes_dispersal': 0,
                      'microbes_mutation': 0,
                    }
    run_rng = {k: np.random.default_rng(d) for k, d in run_seeds.items()}

    if (setup_diagcollector is not None):
        diagcollector = eco.get_diag_collector()
        diagcollector = setup_diagcollector(diagcollector)
    else:
        diagcollector = None

    itime = 0
    hdf_file_path = hdf5_fields_dump_template_str.format(save_dir, itime)
    eco.load_biomass_fields_from_hdf5(hdf_file_path)

    eco, diagcollector = run_simulation_time_loop(
                                                  eco, n_steps, run_rng,
                                                  diagcollector,
                                                  save_fields,
                                                  fields_saving_period,
                                                  save_dir,
                                                  display_runtimes,
                                                  display_steps_messages,
                                                  display_step_period,
                                                  num_threads,
                                                  )
    if (diagcollector is not None):
        diagcollector.dump_all_timelines_to_hdf5(
                hdf5_timelines_dump_template_str.format(save_dir, 0, n_steps)
                                                )
    else:
        print('diagcollector is None')
    return True


def create_simulation_directory(eco,
                                save_dir='.',
                                ):
    """
    From a fully initialized ecosystem object
    save parameters, initial conditions and model functions python
    module in a directory

    Parameters
    ----------
    eco : :class:`dementmutant.ecosystem.Ecosystem`
        Fully initialized (ready to run) `Ecosystem` object
    save_dir : str
        Path of the directory to create and store parameters and intial
        values to.

    """
    os.makedirs(save_dir, exist_ok=True)
    if (save_dir[-1] == '/'):
        save_dir = save_dir[:-1]

    itime = 0
    # set saving tag to fields
    eco.field_locator.add_tags_to_fields(['save', ])
    #
    hdf_file_path = hdf5_fields_dump_template_str.format(save_dir, itime)
    eco.save_biomass_fields_to_hdf5(hdf_file_path)

    json_param_path = json_param_template_str.format(save_dir)
    eco.save_parameters_to_json(json_param_path)
    eco.save_functions_module(save_dir)

    return True


def set_run_seed_in_directory(run_seeds, save_dir='.'):
    """

    Write run seeds in json format in a directory
    This will write the dictionnary containing seeds needed for
    a single replica of a run. This has two main purposes
    - keeping track of which seeds were used for reproductibility
    - preparing and setting up seeds in advancset_run_seed_in_directorye before running simulations.
    (combined with json parameter files, model function modules and hdf5 files
    containing initial data, all data required for performing a single simulation
    is in the directory)

    Parameters
    ----------

    run_seeds : dict of int with keys
        Dictionnary with keys {'microbes_mortality','microbes_dispersal','microbes_mutation'}

    save_dir : str
        Path of the directory where to put seeds

    """
    json_runseeds_path = json_runseeds_template_str.format(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    dict_to_json(run_seeds, json_runseeds_path)


def load_simu_from_dir(save_dir, itime=0,
                       functions_module_name='model_function',
                       functions_module_path=None
                       ):
    """
        Load simulation parameters and state at prescribed time step

    Parameters
    ----------
        save_dir : str
            Path of the simulation directory
        itime: int
            Time index at which we want to recover biomassfields state

    Returns
    -------
        :class:`dementmutant.ecosystem.Ecosystem`
            Fully initialized Ecosystem with state loaded from step `itime`

    """
    json_param_path = json_param_template_str.format(save_dir)
    if (functions_module_path is None):
        sys.path.append(save_dir)
    eco = ecosystem_from_json_file(json_param_path,
                                   functions_module_name,
                                   functions_module_path,
                                   )
    if (itime >= 0):
        eco = reload_fields_state(eco, save_dir, itime)
    return eco


def reload_fields_state(eco, save_dir, itime=0):

    hdf_file_path = hdf5_fields_dump_template_str.format(save_dir, itime)
    eco.load_biomass_fields_from_hdf5(hdf_file_path)

    return eco


def get_simulation_files(save_dir):
    dump_file_pat = hdf5_fields_dump_template_str.format(save_dir, 0).replace('000000', '*')
    dump_files = sorted(glob.glob(dump_file_pat))

    def fname_to_itime(f):
        res = int(f.split('/')[-1].split('.')[0].split('_')[-1])
        return res

    dump_files_dict = {fname_to_itime(f): f for f in dump_files}
    return dump_files_dict
