#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 11:49:52 2022
Copyright CNRS
@author: david.coulette@ens-lyon.fr
"""

import os
import sys
import glob
import resource
sys.path.append('../../')
from examples.commonutils.simutils import run_simu_from_dir
from examples.commonutils.simutils import set_stoechimetry_step_type
from examples.commonutils.simutils import get_stoechimetry_step_type

from refsimu import microbe_pop_on
from refsimu import setup_diagcollector

def _flushout():
    if (not hasattr(sys, 'ps1')):
        sys.stdout.flush()


if __name__ == '__main__':
    print('AAAA')
    # MAX_VIRTUAL_MEMORY = 4000 * 1024 * 1024 # 2000 MB
    # resource.setrlimit(resource.RLIMIT_AS,(MAX_VIRTUAL_MEMORY,resource.RLIM_INFINITY))
    if (len(sys.argv) not in  [3,4]):
        print('Required inputs : either base_dir , number of steps \n or either base_dir , number of steps single task id ( 0 to nb_tasks-1')
        exit()
    base_dir = str(sys.argv[1])
    n_steps = int(sys.argv[2])
    task_id = None
    if (len(sys.argv) == 4):
        task_id = int(sys.argv[3])
    dlist = glob.glob(os.path.join(base_dir,'*/replica_*'))
    dlist += glob.glob(os.path.join(base_dir,'replica_*'))
    dlist = [d for d in dlist if os.path.isdir(d)]
    dlist = sorted(dlist)
    for itask,d in enumerate(dlist):
        print(itask,d)
    task_id_min, task_id_max = 0, len(dlist)-1
    if (task_id is not None):
        if not(task_id >= task_id_min and task_id <= task_id_max):
            print('Wrong task id, should be between {} and {}'.format(task_id_min,task_id_max))
            exit()
    for itask,rep_dir in enumerate(dlist):
        if (task_id is not None and itask != task_id):
            continue
        print(rep_dir)
        seeds = tuple (int(s) for s in os.path.basename(rep_dir).split('_')[1:])
        mod_name = "model_functions_{}_{}_{}".format(*seeds)
        mod_path = os.path.join(rep_dir,"model_functions.py")
        print('Running task {} in directory {}'.format(itask,rep_dir))
        _flushout()
        microbe_pop_on()
        #set_stoechimetry_step_type('standalone') # uncomment to switch to splitted stoechiometry/mortality
        ok = run_simu_from_dir(n_steps=n_steps,
                                save_dir=rep_dir,
                                save_fields=False,
                                fields_saving_period=1,
                                setup_diagcollector=setup_diagcollector,
                                functions_module_name = mod_name,
                                functions_module_path= mod_path
                                )





