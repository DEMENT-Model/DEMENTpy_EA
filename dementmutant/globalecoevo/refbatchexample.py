#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Mon Jan  3 15:08:05 2022

Copyright CNRS

@author: david.coulette@ens-lyon.fr
"""

from refsimu_singletrait import build_and_run_simu

if __name__ == '__main__':

    grid_shape = (64, 64)
    n_degrad_enzymes = 1
    n_taxa = 128
    n_osmolytes = 1
    n_years = 10
    n_steps = 365 * n_years

    save_period = 1
    save_dir =  './simutest_batch'

    num_threads = 4

#    st, eco, diagcollector = build_and_run_simu(grid_shape,
#                                                n_degrad_enzymes,n_taxa,n_osmolytes,n_steps,
#                                                with_diags=False,
#                                                with_fields_saving=False,
#                                                field_saving_period=save_period,
#                                                save_dir=save_dir
#                                                )

    eco, diagcollector = build_and_run_simu(grid_shape,
                                            n_degrad_enzymes,
                                            n_taxa,
                                            n_osmolytes,
                                            n_steps,
                                            with_diags=True,
                                            save_fields=False,
                                            fields_saving_period=save_period,
                                            save_dir=save_dir,
                                            num_threads=num_threads,
                                            save_diags=True
                                            )