#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:17:09 2022
Copyright CNRS
@author: david.coulette@ens-lyon.fr
"""


import os
import sys
import glob
import resource
import multiprocessing
sys.path.append('../../')
from examples.commonutils.simutils import run_simu_from_dir
from examples.commonutils.simutils import set_stoechimetry_step_type
from examples.commonutils.simutils import get_stoechimetry_step_type
from refsimu import microbe_pop_on
from refsimu import setup_diagcollector

# CAVEAT
# There are two switches that can affect behaviour
# nd are not yet included as explicit parameters
# - microbial model quantization, which is set by calling microbe_pop_on
# - separation of stoechiometry correction from death ( starvation/stocahastic)
#   which is activated by calling set_stoechimetry_step_type
#   with either 'standalone' (the new behavior) or 'with_mortality'
#   which is the old and default behavior for now
#
# Those are **temporary** shortcuts during model transition/testing
# In each case, they will either be converted to explicit options
# if both behaviours are meant to be kept, or disappear if only one
# behaviour is selected.
#
# The switch from pure biomass to quantized microbial model has
# visible impact in parameter and datafiles (new parameters for microbes,
# and new biomass field "quanta"). It automatically triggers the use of
# the corresponding operators ( birth, mutation, death, dispersal)
#
# The stoechiometry correction separation option affects only the
# time steps loops and has no impact on parameters or fields existence.
# ( the only clearly visible effect will be in the run logs specifying
#  which operators are called). Be careful when comparing simulations !

def replica_task(rep_dir):
    # MAX_VIRTUAL_MEMORY = 2000 * 1024 * 1024 # 2000 Mb
    # resource.setrlimit(resource.RLIMIT_AS,(MAX_VIRTUAL_MEMORY,resource.RLIM_INFINITY))
    print('WARNING RUNTIME OPTONS')
    microbe_pop_on()
    #set_stoechimetry_step_type('standalone') # uncomment to switch to splitted stoechiometry/mortality
    print('stoechiometry step type {}'.format(get_stoechimetry_step_type()))
    seeds = tuple (int(s) for s in os.path.basename(rep_dir).split('_')[1:])
    mod_name = "model_functions_{}_{}_{}".format(*seeds)
    mod_path = os.path.join(rep_dir,"model_functions.py")
    ok = run_simu_from_dir(n_steps=n_steps,
                            save_dir=rep_dir,
                            save_fields=False,
                            setup_diagcollector=setup_diagcollector,
                            fields_saving_period=1,
                            functions_module_name = mod_name,
                            functions_module_path= mod_path
                            )
    return ok


if __name__ == '__main__':

    if (len(sys.argv) != 4):
        print('3 Required inputs : base_dir , number of steps, number of worker processes')
        exit()
    base_dir = str(sys.argv[1])
    dlist = glob.glob(os.path.join(base_dir,'replica_*'))
    dlist += glob.glob(os.path.join(base_dir,'*/replica_*'))
    dlist = [d for d in dlist if os.path.isdir(d)]
    dlist = sorted(dlist)
    n_steps = int(sys.argv[2])
    n_proc = int(sys.argv[3])

    with multiprocessing.Pool(n_proc) as p:
        chk = p.map(replica_task, dlist)

