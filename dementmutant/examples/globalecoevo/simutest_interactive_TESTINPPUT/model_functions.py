#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various functions for model behaviour

Created on Wed Dec 22 17:13:59 2021
Copyright CNRS
@author: david.coulette@ens-lyon.fr
"""


import numpy as np
import pandas as pd
import io
import itertools

def _linear_thresholded_microbe_osmo_psi(Psi, alpha, Psi_threshold):

    if Psi >= Psi_threshold:
        res = 0.0
    else:
        res = 1 - alpha * Psi
    # ensure bounds
    res = max(res, 0.0)
    res = min(res, 1.0)

    return res


def osmo_psi_modulation(env):
    psi = env['psi']
    alpha = 0.01
    psi_thres = -6.0
    res = _linear_thresholded_microbe_osmo_psi(psi, alpha, psi_thres)
    return res


def _linear_Taxon_AE(temp_celsius):
    AE_ref = 1
    T_ref_celsius = 20.0
    AE_T_slope = -0.005
    res = AE_ref + AE_T_slope * (temp_celsius-T_ref_celsius)
    # ensure assimilation efficiency factor is bounded
    res = max(res, 0.0)
    res = min(res, 1.0)
    return res


def respiration_ae(env):
    T = env['temp']
    AE = _linear_Taxon_AE(T)
    return 1.0-AE


def Allison_wp_factor_substrate(env):
    """
    Ref : Allison & Gouldon 2017 SBB
    """
    wp_fc = 0.0
    rate = 0.05

    psi = env['psi']

    if psi >= wp_fc:
        f_psi = 1.0
    else:
        f_psi = np.exp(rate * (psi - wp_fc))

    return f_psi


def Allison_wp_factor_uptake(env):
    """
    Ref : Allison & Gouldon 2017 SBB
    """
    wp_fc = 0.0
    rate = 0.01

    psi = env['psi']

    if psi >= wp_fc:
        f_psi = 1.0
    else:
        f_psi = np.exp(rate * (psi - wp_fc))

    return f_psi


def _mortality_proba_drought_ajusted_linear(
                                            basal_death_proba,
                                            drought_rate,
                                            drought_tolerance,
                                            psi_threshold,
                                            psi,
                                            ):
    res = 1.0 - (1.0-drought_tolerance) * drought_rate * (psi-psi_threshold)
    res = res * basal_death_proba
    res[res > 1.0] = 1.0
    res[res < 0.0] = 0.0
    return res


def mutation_proba_invariant(microbes, proba,params=None):
    """
        Mutation probability map
        This function returns the identity as a transition probability map
        This is equivalent to no-mutation
        For testing purposes !
    """
    for i in range(proba.shape[0]):
        for j in range(proba.shape[1]):
            if (i == j):
                proba[i, j] = 1.0
            else:
                proba[i, j] = 0
    return proba


def mutation_proba_symshift(microbes, proba, params={'mu':0.0}):
    """
        Mutation probability mao
        Local  jumps [-1, 0, 1] in taxa index space
        mu : mutation rate (jump probability) by direction
        At boundaries invariance probability is 1-mu
        At other points invariance probability is 1-2mu
        This is an example for testing purposes
    """
    mu = params['mu']
    ntax = proba.shape[0]
    proba[()] = 0
    for i in range(ntax):
        im = max(i-1, 0)
        ip = min(i+1, ntax-1)
        proba[i, i] += 1-2.0 * mu
        proba[i, im] += mu
        proba[i, ip] += mu
    return proba


def monomer_leaching_inorg(env, source):
    """
    Quantify leaching of monomers.

    Parameters:
        env:  environnement dictionnary with fields 'temp', 'psi'
    Return:
        rate: scalar; monomers leaching rate
    """

    # Constants
    ref_rate = 0.1  # monomer loss rate for psi = 0.0
    Psi_slope_leach = 0.1  # Moisture sensivity of abiotic monomer loss rate

    rate = np.zeros((source.n_components), dtype=source.mass.dtype)

    for comp_name in ['NH4', 'PO4']:
        ic = source.get_component_index(comp_name)
        rate[ic] = ref_rate * np.exp(Psi_slope_leach * env['psi'])

    return rate


def degenz_decay_func():
        return 0.04

def get_substrate_classes_dict():
    substrate_classes = {
        'LITm':['Starch','Protein1','Protein2','Protein3','OrgP2'],
        'LITs':['Cellulose','Hemicellulose','Lignin','Chitin','OrgP1'],
        'SOCa':['DeadMic','DeadEnz']
    }
    return  substrate_classes

def get_taxon_classes_desc_dict():
    taxon_class_desc = {
    'MICr': {'LITm':(2,2),'LITs':(1,1),'SOCa':(2,1)},
    'MICk': {'LITm':(1,1),'LITs':(2,2),'SOCa':(2,1)},
    }
    return taxon_class_desc

def get_taxonclass_matrix_dict(choose_dict):
    substrate_classes = get_substrate_classes_dict()
    tmp_l= {}
    for k,d in choose_dict.items():
        tmp_l[k] = list(tuple(((vv,d[1]) for vv in v)) for v in itertools.combinations(substrate_classes[k],d[0]))
    tmp_d = {}
    for ib,b in enumerate(itertools.product(*tmp_l.values())):
        tmp_d[ib] = {}
        for bb in b:
            for bbb in bb:
                tmp_d[ib][bbb[0]] = bbb[1]
    return tmp_d


def get_taxon_classes_matrix_dict(classes_desc):
    res = {k:get_taxonclass_matrix_dict(d) for k,d in classes_desc.items()}
    return res


def get_taxon_subs_efficiency_matrix(mic,subs,tax_selection_map):
    tax_mat_d = get_taxon_classes_matrix_dict(get_taxon_classes_desc_dict())
    res = np.zeros((mic.n_taxa,subs.n_components,))
    # assert(all([itax in tax_selection.keys() for itax in range(mic.n_taxa)]))
    for itax in range(mic.n_taxa):
        taxclass,taxid = tax_selection_map(itax)
        cmp_d = tax_mat_d[taxclass][taxid]
        for ksub,v in cmp_d.items():
            isub = subs.get_component_index(ksub)
            res[itax,isub] = v
    return res

def get_tax_subs_efficiency_matrix_dict(mic,subs,tax_selection_map):
    if (not(hasattr(get_tax_subs_efficiency_matrix_dict,'tmd'))):
        tax_mat_d = get_taxon_classes_matrix_dict(get_taxon_classes_desc_dict())
        tmd = {}
        for itax in range(mic.n_taxa):
            taxclass,taxid = tax_selection_map(itax)
            tmd[itax] = tax_mat_d[taxclass][taxid]
        get_tax_subs_efficiency_matrix_dict.tmd = tmd
    return get_tax_subs_efficiency_matrix_dict.tmd

def taxon_selection_map_full(itax):
   it = itax % 100
   if (it < 50):
       return 'MICr',it
   else:
       return 'MICk',it-50

taxon_selection_map = taxon_selection_map_full


def yearly_substrate_input(subst, itime, parameters):
    """
    Provides periodic substrate input with sekectable litter composition
    """

    _litter_compo = {
    'DementDefault': [0.0, 0.0, 146.89, 85.855, 12.21, 5.82774, 48.91425, 12.69704, 12.69704, 12.69704, 12.95847, 3.09412],
    'Desert': [0.0, 0.0, 141.59, 1.85, 20.4, 6.8, 69.25, 13.81, 13.81, 13.81, 15.09, 3.59],
    'Scrubland': [0.0, 0.0, 97.34, 66.53, 37.64, 5.78, 28.36, 16.16, 16.16, 16.16, 12.82, 3.05],
    'Grassland': [0.0, 0.0, 124.76, 42.45, 23.98, 5.69, 38.48, 16.33, 16.33, 16.33, 12.63, 3.0],
    'PineOak': [0.0, 0.0, 108.3, 25.5, 66.9, 7.72, 62.45, 2.65, 2.65, 2.65, 17.12, 4.07],
    'Subalpine': [0.0, 0.0, 113.23, 27.83, 49.18, 7.92, 66.45, 4.56, 4.56, 4.56, 17.56, 4.17],
    'Boreal': [0.0, 0.0, 65.48, 21.91, 16.13, 1.68, 169.62, 6.68, 6.68, 6.68, 4.14, 0.99]
     }

    _substrates_default_stoech = {
        'DeadMic': {'C': 0.0, 'N': 0.0, 'P': 0.0},
        'DeadEnz': {'C': 0.0, 'N': 0.0, 'P': 0.0},
        'Cellulose': {'C': 1.0, 'N': 0.0, 'P': 0.0},
        'Hemicellulose': {'C': 1.0, 'N': 0.0, 'P': 0.0},
        'Starch': {'C': 1.0, 'N': 0.0, 'P': 0.0},
        'Chitin': {'C': 0.85714, 'N': 0.14286, 'P': 0.0},
        'Lignin': {'C': 0.99174, 'N': 0.00826, 'P': 0.0},
        'Protein1': {'C': 0.83484, 'N': 0.16516, 'P': 0.0},
        'Protein2': {'C': 0.83484, 'N': 0.16516, 'P': 0.0},
        'Protein3': {'C': 0.83484, 'N': 0.16516, 'P': 0.0},
        'OrgP1': {'C': 0.96308, 'N': 0.0, 'P': 0.03692},
        'OrgP2': {'C': 0.58763, 'N': 0.25773, 'P': 0.15464}
     }
    steps =[(1.0,'inf'),]
    periodize= False
    default_litter_key = 'DementDefault'
    if ('steps' in parameters.keys()):
        steps = parameters['steps']
    if ('periodize'in parameters.keys()):
        periodize = parameters['periodize']
    if ('default_litter_key' in parameters.keys()):
        default_litter_key = parameters['default_litter_key']

    def set_rescaling_vector(itime,scal_vec):
        i_year = itime // 365
        litter_keys = np.array([a[0] for a in steps])
        tvals = np.array([a[1] if a[1] != 'inf' else np.infty for a in steps])
        stvals = np.cumsum(tvals)
        if (periodize):
            i_year_int = i_year % int(stvals[-1]+1)
        else:
            i_year_int = i_year
            scal_vec[()] = np.array(_litter_compo[default_litter_key])
        for istep,t in enumerate(stvals):
            if (i_year_int <= t):
                scal_vec[()] = np.array(_litter_compo[litter_keys[istep]])
                break
        return scal_vec

    if (not hasattr(yearly_substrate_input, 'buff_dat')):
        bdat = {}
        base_stoech = np.zeros(subst.mass.shape[2:])
        for ksub,d in _substrates_default_stoech.items():
            isub=  subst.get_component_index(ksub)
            for katom,dd in d.items():
                ia = subst.get_atom_index('mass',katom)
                base_stoech[isub,ia] = dd
        res_buff = np.zeros_like(subst.mass)
        scal_vec = np.zeros((subst.n_components,))
        bdat= {
            'base_stoech':base_stoech,
            'res_buff':res_buff,
            'scal_vec':scal_vec
            }
        yearly_substrate_input.buff_dat = bdat

    if (itime >= 0) and ((itime % 365) == 0):
        bdat = yearly_substrate_input.buff_dat
        bdat['scal_vec'] = set_rescaling_vector(itime, bdat['scal_vec'])
        tmp_base = bdat['base_stoech'] * bdat['scal_vec'][:,np.newaxis]
        bdat['res_buff'][()] = tmp_base[np.newaxis,np.newaxis,:,:]
        yearly_substrate_input.buff_dat = bdat
        del(tmp_base)
        return yearly_substrate_input.buff_dat['res_buff']
    else:
        return 0



ref_yearly_climate = """
Day,Temp,FM,Psi,Date
0,10.82317708,51.09261458,-0.699744064,12/15/10
1,11.27633333,59.2778125,-0.590704686,12/16/10
2,9.696135417,65.89679167,-0.52355489,12/17/10
3,12.62658333,69.7083125,-0.49104702,12/18/10
4,12.44277083,70.76352083,-0.482708265,12/19/10
5,12.272,70.93780208,-0.481356542,12/20/10
6,10.80175,70.91869792,-0.481504367,12/21/10
7,12.08052083,70.90282143,-0.481627281,12/22/10
8,10.41860417,70.91040476,-0.481568564,12/23/10
9,12.39826042,70.9179881,-0.481509861,12/24/10
10,10.83905208,70.92557143,-0.481451171,12/25/10
11,10.4173125,70.68268333,-0.483337661,12/26/10
12,10.04410417,70.74476667,-0.482854147,12/27/10
13,12.91078125,70.57978889,-0.484141023,12/28/10
14,10.77119792,70.41481111,-0.485434353,12/29/10
15,7.12040625,70.24983333,-0.486734184,12/30/10
16,5.662729167,70.36344792,-0.485838336,12/31/10
17,7.2355625,70.41664583,-0.485419935,1/1/11
18,7.798333333,70.56305208,-0.484271935,1/2/11
19,9.124427083,69.61684375,-0.491782593,1/3/11
20,10.33377083,70.02628125,-0.48850597,1/4/11
21,12.563375,70.55792708,-0.484312035,1/5/11
22,15.82370833,70.37596875,-0.485739799,1/6/11
23,11.68292708,69.66920833,-0.491361235,1/7/11
24,8.748604167,69.56411458,-0.492207572,1/8/11
25,9.623875,69.42091667,-0.493365181,1/9/11
26,8.7745,69.35880208,-0.493868905,1/10/11
27,12.68341667,67.61225,-0.508438637,1/11/11
28,13.95472917,66.27044792,-0.520190954,1/12/11
29,17.20273958,64.71566667,-0.534461885,1/13/11
30,18.96917708,62.76227083,-0.553466105,1/14/11
31,21.9568125,55.72069792,-0.633882299,1/15/11
32,19.0203125,52.93354167,-0.672069525,1/16/11
33,20.68323958,53.02702083,-0.670719061,1/17/11
34,20.20907292,52.89628125,-0.672609238,1/18/11
35,15.65129167,52.43720833,-0.679326236,1/19/11
36,15.61246875,49.21810417,-0.730205541,1/20/11
37,17.35988542,41.61919792,-0.884041522,1/21/11
38,15.31288542,41.75839583,-0.88068287,1/22/11
39,16.33533333,40.69330208,-0.907008476,1/23/11
40,16.07069792,37.81985417,-0.985977172,1/24/11
41,16.03319792,38.3386875,-0.970780477,1/25/11
42,16.560875,37.9054375,-0.983439761,1/26/11
43,18.40111458,33.83832292,-1.119286902,1/27/11
44,16.51120833,33.26357292,-1.141360786,1/28/11
45,12.65953125,33.61471875,-1.127778677,1/29/11
46,9.85884375,37.19707292,-1.004818206,1/30/11
47,10.07333333,68.63815625,-0.499784398,1/31/11
48,9.413135417,68.31929167,-0.502444462,2/1/11
49,10.53044792,62.74263542,-0.553663567,2/2/11
50,9.741729167,49.52367708,-0.725071445,2/3/11
51,8.29684375,45.55483333,-0.797513791,2/4/11
52,11.07954167,43.30419792,-0.844934979,2/5/11
53,14.96234375,40.91247917,-0.901471248,2/6/11
54,17.07209375,37.49297917,-0.995782627,2/7/11
55,12.60189583,35.91235417,-1.045898333,2/8/11
56,12.89311458,32.69723958,-1.163924598,2/9/11
57,14.12852083,26.25425,-1.494789113,2/10/11
58,15.84677083,23.770625,-1.674099257,2/11/11
59,19.04675,22.01713542,-1.826922654,2/12/11
60,15.92471875,21.80497917,-1.847200417,2/13/11
61,10.96576042,22.60201042,-1.773126939,2/14/11
62,11.63857292,24.0501875,-1.65193303,2/15/11
63,11.87271875,45.21897917,-0.804269926,2/16/11
64,8.5089375,67.94875,-0.505569203,2/17/11
65,8.799697917,66.5728125,-0.517498407,2/18/11
66,9.362604167,69.669375,-0.491359895,2/19/11
67,6.64896875,70.19172917,-0.487193534,2/20/11
68,7.3815625,69.44090625,-0.493203279,2/21/11
69,7.833583333,67.76964583,-0.507092681,2/22/11
70,7.799041667,64.70153125,-0.534594999,2/23/11
71,8.813885417,61.09730208,-0.57069277,2/24/11
72,8.687885417,57.4124375,-0.612633393,2/25/11
73,7.891083333,66.380625,-0.519206791,2/26/11
74,5.263958333,67.94307292,-0.505617361,2/27/11
75,9.2076875,64.61758333,-0.535386822,2/28/11
76,9.288479167,57.1249375,-0.616149567,3/1/11
77,10.12026042,53.85938542,-0.658915161,3/2/11
78,12.6196875,52.72679167,-0.675074579,3/3/11
79,11.85870833,51.78010417,-0.689162694,3/4/11
80,19.08110417,45.72041667,-0.794221947,3/5/11
81,14.25886458,39.18645833,-0.946874526,3/6/11
82,11.0624375,58.06207292,-0.604825366,3/7/11
83,10.84827083,67.36890625,-0.510532816,3/8/11
84,17.530125,63.27665306,-0.548339964,3/9/11
85,17.78058333,54.89096939,-0.644816975,3/10/11
86,14.62384375,49.75386458,-0.721248485,3/11/11
87,12.07852083,44.90677083,-0.810647422,3/12/11
88,11.78588542,40.60342708,-0.909297549,3/13/11
89,13.57873958,37.30630208,-1.001465004,3/14/11
90,15.37309375,34.51116667,-1.094443882,3/15/11
91,14.3015,32.07405208,-1.189740162,3/16/11
92,13.36286458,30.67517708,-1.251786372,3/17/11
93,12.28875,29.33645833,-1.317112074,3/18/11
94,10.99060417,27.86627083,-1.396617873,3/19/11
95,10.34551042,27.81470833,-1.399569749,3/20/11
96,8.413,66.5990625,-0.517265885,3/21/11
97,9.08878125,69.07773958,-0.496160326,3/22/11
98,8.4511875,67.71266667,-0.50757916,3/23/11
99,8.026083333,69.54777083,-0.492339436,3/24/11
100,10.29464583,69.63554167,-0.49163206,3/25/11
101,9.267520833,69.81961458,-0.490154731,3/26/11
102,10.74879167,69.72973958,-0.490875006,3/27/11
103,10.70876042,69.84329167,-0.489965309,3/28/11
104,12.20442708,68.89827083,-0.49763395,3/29/11
105,16.18497917,66.09192708,-0.521793056,3/30/11
106,22.37526042,60.04040625,-0.582159192,3/31/11
107,22.44321875,55.0128125,-0.643189138,4/1/11
108,15.60986458,49.37108333,-0.727626755,4/2/11
109,13.32244792,44.75901042,-0.813698932,4/3/11
110,14.39328125,38.36407292,-0.970048217,4/4/11
111,15.38123958,32.72713542,-1.162712593,4/5/11
112,13.22930208,30.20196875,-1.274169776,4/6/11
113,12.62086458,29.88498958,-1.289587885,4/7/11
114,8.01171875,32.559,-1.169559936,4/8/11
115,6.586197917,45.71095833,-0.794409294,4/9/11
116,10.26535417,48.3536875,-0.745105459,4/10/11
117,12.48523958,37.80753125,-0.986343539,4/11/11
118,13.02959375,30.94892708,-1.239171759,4/12/11
119,12.55960417,27.54317708,-1.415309721,4/13/11
120,12.60920833,25.43388542,-1.549876163,4/14/11
121,18.78069792,22.65440625,-1.768452619,4/15/11
122,21.31609375,21.250125,-1.902284158,4/16/11
123,19.07240625,20.47805208,-1.98425935,4/17/11
124,13.57097917,21.26702083,-1.900561381,4/18/11
125,14.06153125,24.07164583,-1.650254379,4/19/11
126,14.20598958,23.90389583,-1.663463151,4/20/11
127,13.46530208,24.07380208,-1.650085876,4/21/11
128,13.2276875,24.490875,-1.618089668,4/22/11
129,13.61989583,23.46214583,-1.699214778,4/23/11
130,12.48438542,23.52958333,-1.69366401,4/24/11
131,13.48071875,26.10565625,-1.504492501,4/25/11
132,14.65501042,23.68538542,-1.680969249,4/26/11
133,16.62030208,22.08288542,-1.820722902,4/27/11
134,18.73253125,20.70135417,-1.959877411,4/28/11
135,15.02496875,19.87407292,-2.0531489,4/29/11
136,15.40321875,18.62435417,-2.210929507,4/30/11
137,19.44496875,16.34140625,-2.56635979,5/1/11
138,20.7076875,15.25815625,-2.775077668,5/2/11
139,22.99728125,14.1241875,-3.030463992,5/3/11
140,22.72846875,13.06672917,-3.311595124,5/4/11
141,19.85439583,12.85613542,-3.373506793,5/5/11
142,15.79520833,13.4503125,-3.2041481,5/6/11
143,14.36961458,13.51432292,-3.18685274,5/7/11
144,13.22875,15.35275,-2.755594097,5/8/11
145,11.862,17.29673958,-2.40540521,5/9/11
146,12.61357292,16.16776042,-2.597805564,5/10/11
147,13.28517708,13.73318646,-3.12901891,5/11/11
148,14.97097917,12.84458021,-3.376966757,5/12/11
149,16.18344792,12.40424063,-3.513965399,5/13/11
150,13.93976042,14.41380833,-2.961145325,5/14/11
151,11.86318718,29.32465625,-1.317716392,5/15/11
152,10.54705871,53.06921875,-0.670111108,5/16/11
153,11.78623958,41.91930208,-0.876830159,5/17/11
154,12.128875,52.30482292,-0.681286698,5/18/11
155,12.24979167,61.70884375,-0.564249837,5/19/11
156,14.04953125,53.5429375,-0.663356498,5/20/11
157,14.75382292,43.5935625,-0.838544272,5/21/11
158,14.03747917,35.77336458,-1.050532106,5/22/11
159,13.94758333,32.57371875,-1.168957491,5/23/11
160,13.1919375,28.92472917,-1.338506502,5/24/11
161,15.18498958,24.09861458,-1.648149193,5/25/11
162,15.29416667,20.99912604,-1.928226724,5/26/11
163,15.73541667,19.26172188,-2.127722891,5/27/11
164,15.6880625,17.68666875,-2.345044026,5/28/11
165,12.883,19.4970875,-2.098466228,5/29/11
166,14.42835417,21.12222083,-1.915421548,5/30/11
167,15.945125,16.63964271,-2.513988764,5/31/11
168,14.12042708,15.98943021,-2.630860877,6/1/11
169,13.78844792,15.22620625,-2.781716978,6/2/11
170,15.65301042,14.049575,-3.048817688,6/3/11
171,14.61836458,13.36418646,-3.227698858,6/4/11
172,14.24133333,12.84699063,-3.376244459,6/5/11
173,14.84175,22.74749792,-1.760204577,6/6/11
174,15.29942708,24.52348958,-1.615636673,6/7/11
175,15.0995,17.67358125,-2.347023777,6/8/11
176,15.19991667,16.06182708,-2.617346676,6/9/11
177,14.849,16.19931771,-2.592037166,6/10/11
178,15.16461458,18.34497917,-2.249354234,6/11/11
179,15.21665625,17.4021875,-2.38879624,6/12/11
180,15.3701875,15.84718229,-2.657799084,6/13/11
181,17.55060417,13.45166563,-3.203780668,6/14/11
182,17.57588542,12.45727396,-3.496916409,6/15/11
183,16.0864375,12.39342917,-3.517460193,6/16/11
184,15.96213542,12.73275208,-3.410798677,6/17/11
185,16.975625,12.53265,-3.472950312,6/18/11
186,15.18460417,12.04510833,-3.633651555,6/19/11
187,16.90571875,12.94888333,-3.345974649,6/20/11
188,18.29163542,11.33705313,-3.893474023,6/21/11
189,18.62672917,10.86959792,-4.084925331,6/22/11
190,18.17032292,10.71367083,-4.152769515,6/23/11
191,17.8721875,10.56207083,-4.220788031,6/24/11
192,17.4840625,10.34507917,-4.321862399,6/25/11
193,17.14592708,10.15937604,-4.412036209,6/26/11
194,19.88509375,9.813545833,-4.589716914,6/27/11
195,18.17823958,9.782570833,-4.60628778,6/28/11
196,18.18576042,9.88423125,-4.552317933,6/29/11
197,18.49059375,9.634175,-4.687258697,6/30/11
198,21.66436866,9.131429167,-4.982568848,7/1/11
199,22.72776042,8.776745833,-5.212754524,7/2/11
200,21.18040625,8.716045833,-5.254159501,7/3/11
201,22.801,8.98354375,-5.076181305,7/4/11
202,23.6885625,8.749357292,-5.231360821,7/5/11
203,24.29059375,8.899157292,-5.131091533,7/6/11
204,24.47163542,8.411915625,-5.471259247,7/7/11
205,22.90976042,8.391629167,-5.486340087,7/8/11
206,20.36036458,8.700652083,-5.264758247,7/9/11
207,19.5755625,8.792465625,-5.202131361,7/10/11
208,18.84258333,8.821346875,-5.18271948,7/11/11
209,18.79728125,8.879861458,-5.143804239,7/12/11
210,18.1274375,9.036386458,-5.042355033,7/13/11
211,17.243875,9.700879167,-4.650534176,7/14/11
212,18.56308333,10.45060833,-4.272146106,7/15/11
213,18.29816667,9.312107292,-4.8725107,7/16/11
214,18.39873958,8.945442708,-5.100836378,7/17/11
215,20.61884375,8.634583333,-5.310706609,7/18/11
216,22.24634375,8.368059375,-5.503960015,7/19/11
217,20.17544792,8.262477083,-5.584210529,7/20/11
218,19.22202083,8.442285417,-5.448827431,7/21/11
219,19.14915625,8.517080208,-5.394311831,7/22/11
220,19.15177083,8.485382292,-5.417289906,7/23/11
221,19.31903125,8.430276042,-5.457677157,7/24/11
222,22.26598958,8.2752625,-5.574376019,7/25/11
223,22.18821875,7.702642708,-6.049203153,7/26/11
224,18.92314583,8.315134375,-5.543914434,7/27/11
225,18.97139583,8.538867708,-5.37862373,7/28/11
226,19.81088542,8.657097917,-5.294964257,7/29/11
227,19.66038542,8.673445833,-5.283588474,7/30/11
228,19.97808333,10.32410521,-4.331873123,7/31/11
229,22.7629375,11.14106458,-3.971650828,8/1/11
230,25.69838542,8.217015625,-5.619444678,8/2/11
231,23.92911458,7.3063625,-6.424629726,8/3/11
232,21.31830208,7.636161458,-6.109277773,8/4/11
233,18.43534375,7.852485417,-5.91778729,8/5/11
234,19.17009375,8.154682292,-5.668438731,8/6/11
235,18.41227083,8.258658333,-5.587154222,8/7/11
236,18.16428125,8.478855208,-5.422044268,8/8/11
237,18.3774375,8.524138542,-5.389220077,8/9/11
238,18.07072917,8.554263542,-5.367589505,8/10/11
239,18.08388542,8.663502083,-5.290502412,8/11/11
240,18.71719792,8.7108625,-5.257723801,8/12/11
241,19.16817708,8.872704167,-5.148534733,8/13/11
242,20.29420833,8.682771875,-5.27711943,8/14/11
243,20.19604167,8.592422917,-5.340422951,8/15/11
244,19.89529167,8.58915625,-5.342738459,8/16/11
245,19.88148958,8.410184375,-5.472543211,8/17/11
246,21.0654375,8.267007292,-5.580722181,8/18/11
247,18.99104167,8.525702083,-5.388093387,8/19/11
248,18.773875,8.60071875,-5.334551081,8/20/11
249,17.60826042,8.580077083,-5.349183946,8/21/11
250,19.1928125,8.639226042,-5.30745321,8/22/11
251,23.19653125,8.1759125,-5.651662001,8/23/11
252,22.97303125,8.033328125,-5.76615897,8/24/11
253,24.25961458,8.01199375,-5.783665967,8/25/11
254,25.8845,7.651755208,-6.095086471,8/26/11
255,27.20088542,7.443746875,-6.289629575,8/27/11
256,25.40707292,7.57586875,-6.164736309,8/28/11
257,23.906625,7.691384375,-6.059298403,8/29/11
258,19.7155625,8.096407292,-5.714973391,8/30/11
259,19.25282292,8.313314583,-5.545297922,8/31/11
260,18.1955625,8.373997917,-5.499510573,9/1/11
261,18.4020625,8.385896875,-5.490615589,9/2/11
262,19.03222917,8.346305208,-5.520317159,9/3/11
263,19.7683125,8.309657292,-5.548080322,9/4/11
264,21.56957292,9.157059375,-4.966673534,9/5/11
265,28.245,8.58615,-5.344871043,9/6/11
266,29.48534375,7.709366667,-6.043188885,9/7/11
267,28.63413542,7.204564583,-6.528218267,9/8/11
268,20.930875,7.53315,-6.204605126,9/9/11
269,16.69357292,9.368495833,-4.839091557,9/10/11
270,18.08051042,11.25513333,-3.925796236,9/11/11
271,18.04354167,9.243709375,-4.91363312,9/12/11
272,19.98296875,8.911538542,-5.122965396,9/13/11
273,18.84102083,9.181109375,-4.951844575,9/14/11
274,18.35260417,9.335076042,-4.858845931,9/15/11
275,17.05388542,9.778745833,-4.608341854,9/16/11
276,16.59625,10.58875521,-4.208664368,9/17/11
277,17.44561458,10.25465313,-4.365335003,9/18/11
278,19.10660417,9.555389583,-4.731341703,9/19/11
279,17.55567708,9.56965,-4.723304962,9/20/11
280,18.00876042,9.630571875,-4.689257925,9/21/11
281,17.67958333,9.699089583,-4.651512391,9/22/11
282,18.8746875,9.813267708,-4.589865206,9/23/11
283,17.52566667,9.9676125,-4.508930925,9/24/11
284,16.81322917,10.41250729,-4.289971683,9/25/11
285,16.79,10.60105104,-4.20309991,9/26/11
286,19.36422917,10.27985,-4.353139274,9/27/11
287,17.91569792,10.04360833,-4.470057943,9/28/11
288,18.03828125,10.14373542,-4.419792372,9/29/11
289,18.49536458,9.867525,-4.561105317,9/30/11
290,21.71314583,9.594580208,-4.709316444,10/1/11
291,23.54121875,8.787489583,-5.205489684,10/2/11
292,20.25810417,8.849384375,-5.164004371,10/3/11
293,14.76917708,10.94185521,-4.054187126,10/4/11
294,13.68634375,29.05761458,-1.331530551,10/5/11
295,13.07382292,69.63523958,-0.491634492,10/6/11
296,13.63055208,68.32172917,-0.502424027,10/7/11
297,16.51525,63.44096875,-0.546721194,10/8/11
298,20.06117708,55.01354167,-0.643179419,10/9/11
299,19.02747917,45.60860417,-0.796442007,10/10/11
300,19.50598958,40.05195833,-0.923584014,10/11/11
301,27.77304167,35.05651042,-1.075056253,10/12/11
302,30.6724375,29.12360417,-1.328091668,10/13/11
303,24.11671875,26.39286458,-1.485842735,10/14/11
304,17.28535417,26.64660417,-1.469723899,10/15/11
305,16.33034375,27.0704375,-1.443520233,10/16/11
306,19.80811458,26.93663542,-1.451697309,10/17/11
307,16.75895833,26.44685417,-1.482385319,10/18/11
308,15.54357292,26.98707292,-1.448604722,10/19/11
309,14.6809375,26.99925,-1.447859934,10/20/11
310,15.74436458,26.30189583,-1.491702604,10/21/11
311,19.03621875,25.83427083,-1.522522822,10/22/11
312,20.30598958,24.99745833,-1.580761013,10/23/11
313,14.97155208,24.85190625,-1.591319641,10/24/11
314,14.23173958,26.03280208,-1.509293298,10/25/11
315,15.19127083,28.51508333,-1.360449399,10/26/11
316,16.43934375,26.77444792,-1.4617264,10/27/11
317,17.6443125,24.1070625,-1.647490783,10/28/11
318,18.50957292,22.63461458,-1.770215546,10/29/11
319,20.24051042,21.7704375,-1.85054193,10/30/11
320,20.21560417,20.98017708,-1.930212208,10/31/11
321,15.12838542,20.83823958,-1.945207422,11/1/11
322,19.26320833,19.75219792,-2.067597032,11/2/11
323,18.43264583,17.09484167,-2.437818084,11/3/11
324,12.46460417,31.71257917,-1.205212161,11/4/11
325,9.290072917,64.94671875,-0.53229485,11/5/11
326,9.147083333,65.69454167,-0.525392781,11/6/11
327,10.28155208,68.91935417,-0.497460408,11/7/11
328,11.87153125,68.55635417,-0.500464291,11/8/11
329,15.69853125,67.00171875,-0.513723595,11/9/11
330,18.96708333,62.80204167,-0.553066558,11/10/11
331,17.7365625,58.20889583,-0.603086514,11/11/11
332,13.59754167,57.5099375,-0.611449491,11/12/11
333,14.59377083,65.52863542,-0.526909474,11/13/11
334,13.74592708,65.60910417,-0.526172814,11/14/11
335,13.69016667,64.61183333,-0.535441139,11/15/11
336,12.47923958,63.01382292,-0.550948043,11/16/11
337,15.81257292,61.11439583,-0.570510803,11/17/11
338,10.60483333,58.79369792,-0.596252761,11/18/11
339,11.65492708,57.69389583,-0.609227419,11/19/11
340,10.34760417,56.64097917,-0.622154764,11/20/11
341,10.00734375,69.17765278,-0.495343481,11/21/11
342,11.13190625,68.70109375,-0.499262476,11/22/11
343,12.64646875,68.30854167,-0.502534605,11/23/11
344,11.83244792,67.68514583,-0.507814442,11/24/11
345,10.92446875,67.00060417,-0.513733337,11/25/11
346,16.98958333,65.99902083,-0.522630496,11/26/11
347,23.6,62.54121875,-0.555696755,11/27/11
348,21.634625,57.22947917,-0.614866631,11/28/11
349,18.759875,53.67797917,-0.661454338,11/29/11
350,17.00969792,51.32917708,-0.696068827,11/30/11
351,11.92963542,49.0098125,-0.733744438,12/1/11
352,13.15970833,46.23696875,-0.784114752,12/2/11
353,10.5231875,44.63470833,-0.816282732,12/3/11
354,11.08254167,42.5336875,-0.862406116,12/4/11
355,9.850479167,40.62701042,-0.908695845,12/5/11
356,11.91814583,37.58547917,-0.992989336,12/6/11
357,12.444625,35.507375,-1.0595082,12/7/11
358,11.60592708,34.25380208,-1.103823091,12/8/11
359,12.68482292,33.92189583,-1.116143813,12/9/11
360,15.67677083,32.48695833,-1.172517059,12/10/11
361,9.491416667,31.84633333,-1.199443323,12/11/11
362,9.020614583,38.64472917,-0.962021068,12/12/11
363,9.20034375,65.80447727,-0.524392272,12/13/11
364,10.071625,66.05309091,-0.522142812,12/14/11
"""


def env_func_constant(itime):
    return {'temp': 15.7, 'psi': -0.1}

def env_func_perio(itime):
    if not(hasattr(env_func_perio, 'env_dict')):
        # on the first call
        # reads the data as a pseudo-file from a string
        # and stores values in a dictionnary
        cl_df = pd.read_csv(io.StringIO(ref_yearly_climate),
                            index_col=0,
                            dtype={k: np.float64 for k in ['Temp', 'FM', 'Psi']}
                            )
        env_func_perio.env_dict = {k: d for k, d in cl_df.to_dict().items()
                                   if k in ['Temp', 'Psi']
                                   }
    i = itime % 365
    return {
            'temp': env_func_perio.env_dict['Temp'][i],
            'psi': env_func_perio.env_dict['Psi'][i]
            }
