#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 11:07:22 2022
Copyright CNRS
@author: david.coulette@ens-lyon.fr
"""

import sys
import os
import itertools
from refsimu import build_simu_replica
from refsimu import microbe_pop_on

if __name__ == '__main__':

    microbe_pop_on()
    if (len(sys.argv) != 2):
        print('1 Required input : path base directory to create')
        exit()

    print('Starting preparation of simulation tree')

    #grid_shape = (64, 64)
    grid_shape = (30, 30)
    n_degrad_enzymes = 12
    n_taxa = 100
    n_osmolytes = 1

    litter_names =['Desert','Scrubland','Grassland','PineOak','Subalpine','Boreal']

    subst_descs_single = { '{}_{}'.format(k.upper(),typ):{'name':k.upper(),'steps':[(k,'inf')],'type':typ}
                              for k in litter_names
                              for typ in ['Yearly','Daily']
                          }
    
    # Single substrate with boreal (code Elsa)
    '''
    subst_descs_single = { k.upper():{'name':k.upper(),'steps':[(k,'inf')]}
                              for k in litter_names
                          }
    subst_descs_selected = subst_descs_single
    '''

    def get_dual_sub_desc(k1,a1,k2,a2,src_type):
        res = {
            'name': '{}_{}_{}_{}'.format(k1.upper(),a1,k2.upper(),a2),
            'steps':[(k1,a1),(k2,a2)],
            'periodize':True,
            'type': src_type,
            }
        return res

    tlitter = 50
    subst_descs_dual = {}
    for k1,k2,typ in itertools.product(litter_names,litter_names,['Yearly', 'Daily']):
        tmp_r = get_dual_sub_desc(k1, tlitter, k2,tlitter,typ)
        subst_descs_dual[tmp_r['name']] = tmp_r

    # subst_descs_selected = subst_descs_single
    subst_descs_selected = subst_descs_dual


    opt_d = {
    'substrate' :   subst_descs_selected.values(),
    'mut': [(False,None),(True,0.05)],
    'init_div': [('uniform',None),('dirac',5)],
    'boundary_resampling' : [False,True],
    }
    # mut_options = ['Nomut','Mut'][:1]
    opt_all = itertools.product(*(opt_d.values()))

    def opt_to_dic(t):
        return {k:d for k,d in zip(opt_d.keys(),t)}
    def opt_filt_in(t):
        d = opt_to_dic(t)  # convert option tuple to dict for convenience
        res = True # by default accept all
        # add inclusion conditions eg
        #res = res and  d['mut][0] == True
        #
        # example : select only yearly input
        # res = res and d['substrate']['type'] == 'Yearly'

        return res
    def opt_filt_out(t):
        d = opt_to_dic(t)
        res = False  # by default reject nothing
        res = res or (d['mut'][0] == True) # reject mutation case
        res = res or (d['init_div'][0] == 'dirac') # reject single taxon case
        res = res or (d['boundary_resampling'] == True)  # reject resampling of boundary fluxes
        # add exclusion condtion
        #res = res or d['mut][0] == True
        return res
    def opt_filt(t):
        res = opt_filt_in(t) and not(opt_filt_out(t))
        return res
    opt_filtered = filter(opt_filt,opt_all)

    #mort_seeds = [0,42,133,9999,12,3,19,23]
    #disp_seeds = [5,4234,40,2]
    #mut_seeds = [0,13]
    mort_seeds = [42,]
    disp_seeds = [5,]
    mut_seeds = [0,]
    # mort_seeds = [0,42,]
    # disp_seeds = [5,4234]
    # mut_seeds = [13,57454]
    base_run_seeds = {
                'microbes_mortality': 0,
                'microbes_dispersal': 0,
                'microbes_mutation': 0,
                }
    sds = set(s for s in itertools.product(mort_seeds,disp_seeds,mut_seeds))
    print(sds)

    #
    root_dir = str(sys.argv[1])
    print('Base dir {}'.format(root_dir))
    os.makedirs(root_dir,exist_ok=True)
    #
    for s in sds:
        run_seeds = {k:v for k,v in zip(base_run_seeds.keys(),s)}
        for op_tup in opt_filtered:
            op_dic = opt_to_dic(op_tup)
            print('Building simu for {}'.format(op_dic))
            case_parameters = {
                            'substrate_inputs': op_dic['substrate'],
                            'initial_diversity': op_dic['init_div'][0],
                            'selected_tax': op_dic['init_div'][1],
                            'mutation_rate':op_dic['mut'][1],
                            'mutation_type': 'two', # set to 'two' for 2 mutating daughters
                            'disp_proba0': 0.8,
                            'boundary_flux': op_dic['boundary_resampling'], # set to true to reinject different taxa
                            'diagnostic_period_day':1
                            }
            build_simu_replica(grid_shape,
                                n_degrad_enzymes,
                                n_taxa,
                                n_osmolytes,
                                param_seeds=None,
                                init_seeds=None,
                                run_seeds=run_seeds,
                                root_dir=root_dir,
                                case_parameters=case_parameters
                                )
