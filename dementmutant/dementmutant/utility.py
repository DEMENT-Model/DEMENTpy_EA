#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============
Utility module
==============


"""
import numpy as np
from scipy.stats import distributions
import timeit
import json


class StageTimer:
    """
        A small utility class to record execution runtimes
    """
    def __init__(self):
        self.tstart = timeit.default_timer()
        self.tlast = self.tstart
        self.events = {}

    def tag_event(self, event_name):
        tnew = timeit.default_timer()
        self.events[event_name] = tnew-self.tlast
        self.tlast = tnew

    def set_end(self, evname='cycle'):
        self.tend = timeit.default_timer()
        self.total_duration = self.tend-self.tstart

    def display(self, show_stages=True):
        td = sum(self.events.values())
        if (show_stages):
            for k, d in self.events.items():
                print('{}: {} s ({:.2f}%)'.format(k, d, 100.0 * d / td))
        if (hasattr(self, 'total_duration')):
            print('Total {} {}'.format(td, self.total_duration))
        print('*'*20)


def dict_from_json(filename):
    with open(filename, 'r') as f:
        res = json.load(f)
    return res


def dict_to_json(pdict, filename):
    with open(filename, 'w') as f:
        json.dump(pdict, f)

# from dementpy original code at https://github.com/bioatmosphere/DEMENTpy
# Copyright (c) 2020 Bin Wang
# MIT License
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def LHS(n, loc, upc, dist, rng):
    """
    Latin hypercube sampling.

    Parameters:
        n:    integer; size of desired sampling
        loc:  scalar; lower bound of desired distribution
        upc:  scalar; upper bound of desired distribution
        dist: string; either 'uniform' or 'normal'
        rng : instance of numpy.random generator
    Returns:
        lhs: 1D array
    """

    lower_limits = np.arange(0, n) / n
    higher_limits = np.arange(1, n+1) / n

    points = rng.uniform(low=lower_limits, high=higher_limits, size=n)
    rng.shuffle(points)

    scale = upc - loc
    if dist == 'uniform':
        rv = distributions.uniform(loc=loc, scale=scale)
    elif dist == 'normal':
        rv = distributions.norm(loc=loc, scale=scale)

    lhs = rv.ppf(points)

    return lhs


def _draw_array_from_bounds_lhs(a, bounds, rng):

    if (bounds[1] == bounds[0]):
        a[()] = bounds[0]
    else:
        a[()] = LHS(a.size,
                    bounds[0],
                    bounds[1],
                    'uniform',
                    rng
                    ).reshape(a.shape)
    return a


def _draw_df_from_bounds_lhs(df, bounds, rng):
    df.values = _draw_array_from_bounds_lhs(df.values, bounds, rng)
    return df


def _draw_array_from_array_linear_constraint(a, a_ref,
                                             relative_error,
                                             slope,
                                             intercept,
                                             bounds,
                                             rng
                                             ):
    sigma = np.mean(a_ref) * relative_error
    res = np.abs(rng.normal(a_ref * slope, sigma) + intercept)
    if (bounds[0] is not None):
        res[res < bounds[0]] = bounds[0]
    if (bounds[1] is not None):
        res[res > bounds[1]] = bounds[1]
    a[()] = res[()]
    return a


def _draw_df_from_df_linear_constraint(df, df_ref,
                                       rel_variance,
                                       slope,
                                       intercept,
                                       bounds,
                                       rng,
                                       ):
    df.values = _draw_array_from_array_linear_constraint(
                df.values,
                df_ref.values,
                slope,
                intercept,
                bounds,
                rng
            )
    return df
