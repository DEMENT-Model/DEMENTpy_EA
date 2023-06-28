#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base data structure for biomass concentration fields

Created on Fri Oct 22 14:40:01 2021

Copyright CNRS

author: David Coulette david.coulette@ens-lyon.fr

-------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from dementmutant.defaults import _default_dtype, _biomass_atoms_names
from dementmutant.operators import _update_mass_ratios
from dementmutant.operators import _update_biomass_fields


class BiomassField:
    """
        BiomassField Class

        This class is meant to store and manipulate multicomponent biomass pools data.

        Biomass concentrations are stored a (nx, ny, nc, na) C-ordered numpy ndarrays.
        with

            - (nx,ny) : grid shape
            - nc  : number of components (taxa, substrate, whatever..)
            - na the number of elements (atoms) concentrations fields.
              (usually na=3, for C,N,P content)

        The core field is the mass field.

        Derived fields are

        - tmass : mass summed over atomic components
        - ratios : stoechiometry (mass /tmass)

        Optionnaly additionnal "delta" buffer can be reserved to serve as a duplicate
        temporary storage by computational routines.

        Optionally an additional quanta buffer with one component per taxon
        is reserved to keep track of the number of quantas per component
        for pseudo-quantized fields. Though this field values are in essence
        integers, it is allocated as a float for uniformity. Proper interpretation
        as integers is up to routines using this field.


        Various routines are provided to set/get values and compute
        derived quantities.



    """
    __space_ndims = 2  # 2d grid for now
    __space_axes = (0, 1)
    __axes_names = ['x', 'y', 'Component', 'Atom']

    #: :class:`numpy.ndarray` of float, shape (nx,ny,nc,na)
    mass = None
    #: :class:`numpy.ndarray` of float, shape (nx,ny,nc,1)
    tmass = None
    #: :class:`numpy.ndarray` of float, shape (nx,ny,nc,na)
    ratios = None
    #: :class:`numpy.ndarray` of float, shape (nx,ny,nc,na), optional
    delta_mass = None
    #: :class:`numpy.ndarray` of float, shape (nx,ny,nc,1), optional
    delta_tmass = None
    #: :class:`numpy.ndarray` of float, shape (nx,ny,nc,na), optional
    delta_ratios = None
    #: 2-tuple of int
    grid_shape = None
    #: boolean
    is_quantized = False
    #: :class:`numpy.ndarray` of float, shape (nx,ny,nc,1), optional
    quanta = None
    #: int
    quantization_atom_id = None
    # :class:`numpy.ndarray` of float, shape(nc)
    quantization_norm = None

    def __init__(self,
                 grid_shape,
                 component_names,
                 biomass_names=_biomass_atoms_names,
                 biomass_dtype=_default_dtype,
                 delta_buffers=[],
                 is_quantized=False,
                 ):
        """
        Parameters
        ----------
            grid_shape: tuple of int
                (nx,ny) shape of 2d space grid
            component_names : list of str
                List of component names (taxa, substrates...) as strings. The number
                of components is infered from this list
            biomass_names: list of strings (optionnal)
                Element names for the atom axis - default is (C,N,P). This sets
                the number of atomic components of the biomass field, ie the last dimension
                of the ndarrays.
            delta_buffer: list of str, optional
                List of attributes (mass, tmass, ratios) for which a duplicate
                "delta" buffer (:class:`numpy.ndarray`) should be allocated.

        """
        self.__biomass_atoms_names = biomass_names
        self.__biomass_dtype = biomass_dtype
        self.__biomass_n_atoms = len(self.__biomass_atoms_names)
        self.__delta_buffers = delta_buffers

        assert(len(grid_shape) == 2)
        self.grid_shape = grid_shape
        self.names = component_names
        self.__component_indexes = {k: i for i, k in enumerate(self.names)}
        self.n_components = len(self.names)
        self.grid_size = np.prod(np.array(self.grid_shape, dtype=np.int_))
        self.is_quantized = is_quantized

        default_axes_sizes = {
                              'x': self.grid_shape[0],
                              'y': self.grid_shape[1],
                              'Component': self.n_components,
                              'Atom': self.__biomass_n_atoms,
                             }

        self.__fields_desc = {
                'mass': {'x': None, 'y': None, 'Component': None, 'Atom': None},
                'ratios': {'x': None, 'y': None, 'Component': None, 'Atom': None},
                'tmass': {'x': None, 'y': None, 'Component': None, 'Atom': 1},
                }
        if (self.is_quantized):
            self.__fields_desc['quanta'] ={'x': None, 'y': None, 'Component': None, 'Atom': 1}
            self.quantization_atom_id = 0 # default is carbon
            self.quantization_norm = np.ones((self.n_components,))

        for fname, d in self.__fields_desc.items():
            for ax, dim in d.items():
                if (dim is None):
                    self.__fields_desc[fname][ax] = default_axes_sizes[ax]

        self.__atom_names = {k: self.__biomass_atoms_names for k in ['mass', 'ratios']}
        self.__atom_names['tmass'] = ['+'.join(self.__biomass_atoms_names), ]
        if self.is_quantized:
            self.__atom_names['quanta'] = ['n' ,]

        self.__atom_indexes = {k1: {k: i for i, k in enumerate(d)}
                               for k1, d in self.__atom_names.items()
                               }
        self.__axes_ids = {}
        for fname, dims in self.__fields_desc.items():
            fshape = tuple(dims.values())
            self.__axes_ids[fname] = {k: i for i, k in enumerate(list(dims.keys()))}
            setattr(self, fname, np.zeros(fshape, dtype=self.__biomass_dtype))
            if (fname in delta_buffers):
                setattr(self, 'delta_{}'.format(fname),
                        np.zeros(fshape, dtype=self.__biomass_dtype)
                        )

    def get_biomass_dtype(self):
        """

            Return the numerical type of allocated buffer


        """
        return self.__biomass_dtype

    def get_space_axes(self):
        """

            Return indexes of spatial axes of fields as a tuple

            Return
            ------
                tuple of int
                    tuple of indexes of space axes of data (mass, tmass, ratios) ndarrays

        """
        return self.__space_axes

    def get_space_axes_names(self):
        """
            Return spatial axes names

            Return:
                iterable
                space axes names
        """
        return (self.__axes_names[i] for i in self.__space_axes)

    def get_component_index(self, componentname):
        """

            Return index of component from its name

            Parameters
            ----------
                componentname : string
                    Name of component

            Return
            ------
                int
                    Index of component

        """
        if (componentname in self.__component_indexes.keys()):
            return self.__component_indexes[componentname]
        else:
            print('Unknown component')
            return None

    def is_valid_atomname(self, fieldname, atomname):
        """

            Checks if an atom (element) name is valid for a given field

            Parameters
            ----------

                fieldname : string
                    name of field
                atomname : string
                    atom name (element)

            Return
            ------
            bool
                True if field contains atomname, else False

        """
        return (atomname in self.__atom_indexes[fieldname].keys())

    def get_atomnames(self, fieldname):
        """

            Get valid atomname for a given field


            Parameters
            ----------

                fieldname : str
                    name of the field

            Return
            ------
            list
                A list of valid atom name for the field

        """
        return list(self.__atom_indexes[fieldname].keys())

    def get_atom_index(self, fieldname, atomname):
        """
            Return index of atom along atomic axe of field

            Parameters
            ----------

                fieldname :string
                    name of the field
                atomname : string
                    atom name

            Return
            ------
                int
                    index of atom along atomic axe if atomname is valid
        """
        assert(fieldname in self.__atom_indexes.keys())
        if (atomname in self.__atom_indexes[fieldname].keys()):
            return self.__atom_indexes[fieldname][atomname]
        else:
            print('Unknown atom')
            return None

    def update_biomass_fields(self):
        """

        Computes :

            - total mass concentration across constituants
            - biomass ratios fields, ie relative weight of constituants.


        :todo:

            - change name, this is ambiguous

        """

        self.tmass, self.ratios = _update_biomass_fields(self.mass, self.tmass, self.ratios)

    def get_standard_quantization(self):
        """
        Compute number of quanta per component using a uniform quantization
        rule base on a nomimal value. The underlying model is that of
        quanta whose biomass is equal to a nominal per compopent value
        plus an additional positive fluctuation uniform across quanta and bounded
        by the nominal value.
        This form maximize the number of quanta for a given biomass content
        and minimize per quanta biomass deviation from the nominal value.

        Returns
        -------
        res : `class:numpy.ndarray` of float
            A field of number of quanta expressed as floats.

        """
        # TODO add option of quantization on total mass
        if (not self.is_quantized):
            return None
        res = np.floor_divide(self.mass[:,:,:,self.quantization_atom_id],
                              self.quantization_norm[np.newaxis,np.newaxis,:]
                             )
        return res

    def set_standard_qanta(self):
        if (not self.is_quantized):
            return
        self.quanta[()] = self.get_standard_quantization()

    def update_quantization(self, constraint='up'):
        """
        Compute new quantization and store the difference with previous one
        in a delta buffer

        Parameters
        ----------
        constraint : str, optional, in ['up','down','none']
            Filter on number of quanta evolution. 'up' only allows
            increase,'down' only allows decrease, 'none' applies no filter
            The default is 'none'.

        Returns
        -------
        None.

        """
        # TODO : add option for quantization on total mass
        if (not self.is_quantized):
            return


        ffuncs = {
            'up': np.maximum,
            'down': np.minimum,
            'none': lambda x,y : y
            }
        assert(constraint in ffuncs.keys())

        self.delta_quanta[:,:,:,0] = (
            ffuncs[constraint](
                                    self.quanta[:,:,:,0],
                                    self.get_standard_quantization()
                                )
                                - self.quanta[:,:,:,0]
                                )

        # self.delta_quanta[:,:,:,0] = np.maximum(
                 # self.quanta[:,:,:,0],
                 # self.get_standard_quantization()) - self.quanta[:,:,:,0]
        self.quanta += self.delta_quanta

    def set_fields_from_dict(self, field_dict):
        """
            Set values of biomass mass field from a dictionnary

            Parameters
            ----------

                field_dict : dict
                    A dictionnary of ndarrays of shape (nx,ny,natoms), keyed by component.
                    Not all components need to be referred to.

        """
        for k, d in field_dict:
            if (k in self.names):
                i = self.component_indexes[k]
                self.mass[:, :, i, :] = d
            else:
                ('WARNING : {} component name unknown - ignoring'.format(k))
        self.update_biomass_fields()

    def get_biomass_df_from_file(self, f):
        res = pd.read_csv(f,
                          index_col=0,
                          dtype={k: np.float64 for k in self.__biomass_atoms_names})
        return res

    def set_uniform_biomass_fields_from_df(self, biomass_values_df):
        """
            Set constant in space values for mass biomass field from a pandas Dataframe

            Parameters
            ----------

                biomass_values_df : pandas.DataFrame
                    a DataFrame with component names as index and atom (element) names as columns

        """
        for k in biomass_values_df.index:
            if (k in self.__component_indexes.keys()):
                ik = self.__component_indexes[k]
                for c in biomass_values_df.columns:
                    if (c in self.__atom_indexes['mass'].keys()):
                        ic = self.__atom_indexes['mass'][c]
                        self.mass[:, :, ik, ic] = biomass_values_df.loc[k, c]
                    else:
                        print('WARNING unkown atom name {} - ignoring'.format(c))
            else:
                print('WARNING : {} component name unknown - ignoring'.format(k))
        self.update_biomass_fields()

    def set_random_masses_from_bounds(self, bounds, rng=np.random, stoechio=None, ref_scale='T'):
        """

            Set biomass values using a uniform distribution with compact support
            By default both total biomass and sotechiometry are random. Stoechiometry
            can be imposed optionnaly (per component or for all components),
            and in that case only total mass is randomly drawn.

            Parameters
            ----------

                bounds : tuple of floats
                    Lower and higher bounds of the uniform distribution
                rng  : numpy.random generator
                    Random generator to use
                stoechio : array-like (optional)
                    Imposed stoechiometry. This can be either
                        - a common stoechimetry vector of shape (n_atoms,) for all components
                        - a vector of shape (n_components, n_atoms) for per component stoechiometry

                ref_scale : str in ['T','C','N','P']
                    reference atomic component for the bounds (default is T for total mass)

        """
        if (stoechio is None):
            draws = rng.uniform(bounds[0], bounds[1], size=self.mass.shape)
            self.mass[()] = draws[()]
            self.update_biomass_fields()
        else:
            stoech = np.array(stoechio).astype(self.ratios.dtype)
            assert(stoech.shape[-1] == self.mass.shape[-1])
            if (len(stoech.shape) == 1):
                stoech = np.repeat(stoech[np.newaxis, :], self.n_components, axis=0)
            elif(len(stoech.shape) == 2):
                assert(stoech.shape[0] == self.n_components)
            else:
                print('Shape must be (n_atoms,) or (n_comps,n_atom) - ignoring')
            draws = rng.uniform(bounds[0], bounds[1], size=self.mass[:, :, :, 0].shape)
            if (ref_scale == 'T'):
                truc = np.sum(stoech, axis=-1, keepdims=True)
            else:
                ia = self.get_atom_index('mass', ref_scale)
                truc = np.expand_dims(stoech[:,ia],axis=1)
            scal = np.repeat(truc, stoech.shape[-1], axis=-1)
            scal[scal == 0.0] = 1.0
            stoech = stoech / scal
            for ia in range(self.mass.shape[-1]):
                self.mass[:, :, :, ia] = (
                        stoech[np.newaxis, np.newaxis, :, ia]
                        *
                        draws[:, :, :]
                        )

            self.update_biomass_fields()

    def get_component_fields_dict(self, fieldname, componentname, atom_names=None):
        """
            Get a per-element (atom) dictionnary of a single component of a single field.

            Parameters
            ----------

                fieldname : string
                    name of the field ( mass ,ratios, tmass)
                componentname : string
                    the required component
                atom_names iterable of strings(optional)
                    element (atom) selection if None, all elemental fields are returned

            Returns
            -------
                dict
                    A dictionnary of 2d ndarrays, keyed by element (atom) name

        """
        assert(fieldname in self.__fields_desc.keys())
        res = {}
        if (atom_names is None):
            atom_names = [c for c in self.__atom_names[fieldname]]
        if (isinstance(atom_names, str)):
                atom_names = [atom_names, ]
        f = getattr(self, fieldname)
        cid = self.__component_indexes[componentname]
        res = {}
        for a in atom_names:
            if (a in self.__atom_names[fieldname]):
                res[a] = f[:, :, cid, self.__atom_indexes[fieldname][a]]
            else:
                print('Warning unknown atom names {}'.format(a))
        return res

    def get_space_average(self, fieldname='mass'):
        """

            Get a per-component field averaged over spatial grid

            Parameters
            ----------

                fieldname : string
                    name of required field (mass, ratios, tmass)

            Returns
            -------
                numpy.ndarray
                    A 2d (n_component, n_atoms) ndarray of the space averaged field

            :todo:
                factorize with sum

        """
        f = getattr(self, fieldname)
        res = np.mean(f, axis=self.__space_axes)
        return res

    def get_space_total(self, fieldname='mass'):
        """

            Get a per-component per-atom (element) sum over space of a field

            Parameters
            ----------

                fieldname : string
                    name of required field (mass, ratios, tmass)

            Returns
            -------
                numpy.ndarray
                A 2d (n_component, n_atoms) ndarray of the space summed field

        """
        f = getattr(self, fieldname)
        res = np.sum(f, axis=self.__space_axes)
        return res

    def get_local_field_df(self, fieldname, coords):
        """

            Get field values in a given cell of the grid as a pandas DataFrame

            Parameters
            ----------

                fieldname :string
                    name of required field (mass, ratios, tmass)
                coords : tuple of ints
                    (row, col) coordinates of the cell.


            Returns
            -------
                pandas.Dataframe
                    A dataframe with components as indexes and element names
                    as columns
        """
        assert(len(coords) == self.__space_ndims)
        f = getattr(self, fieldname)
        if (fieldname in ['mass', 'ratios']):
            dat = f[coords[0], coords[1], :, :]
            columns = self._atom_names[fieldname]
        index = self.names
        df = pd.DataFrame(data=dat, index=index, columns=columns, copy=True)
        return df

    def get_space_total_df(self, fieldname='mass'):
        """

            Get per-component values of field summed over spatial grid
            as a Pandas dataframe.

            Parameters
            ----------

                fieldname : string
                    name of the required field

            Return
            ------
                pandas.DataFrame
                A pandas dataframe (row : components, cols : atomic concentration)

        """
        dat = self.get_space_total(self, fieldname)
        index = self.names
        columns = self._atom_names[fieldname]
        df = pd.DataFrame(data=dat, index=index, columns=columns, copy=True)
        return df

    def check_finiteness(self):
        """

            Check for finiteness of field attributes



            Return
            -------
                dict
                 A dictionnary of boolean (True : finite False : not)
                 keyed by field names ( mass, tmass, ratios)

        """

        res = {k: np.all(np.isfinite(getattr(self, k)))
               for k in self.__fields_desc.keys()
               }
        return res

    def _save_to_hdf_file(self, fieldname, hdf_file, saved_data=['mass', ]):
        """
            Add a group to an openend hdf file and fill it with required data

            Parameters
            ----------

                fieldname : str
                    The name of the biomassfield
                    (an error will occur if a non-group object with the same name already exists)
                hdf_file : h5py.File
                    An instance of a h5py.File object, opened with 'a' attribute
                save_data: list of str
                    list of names of attributes to save (by default only mass)
        """
        grp = hdf_file.require_group(fieldname)
        # add_data = { False:[] , True:['quanta','quantization_norm']}[self.is_quantized]
        add_data = { False:[] , True:['quanta',]}[self.is_quantized]

        for a in saved_data+add_data:
            if (a in grp.keys()):
                del(grp[a])
            grp.create_dataset(name=a, data=getattr(self, a), compression='gzip')

    def _load_from_hdf_file(self, fieldname, hdf_file, saved_data=['mass', ]):
        """
            Loads data buffer from hdf5 file

            Parameters
            ----------

                fieldname : str
                    Name of the hdf group where data was saved
                hdf_file : h5py.File
                    An instance of and opend h5py.File object from which to read
                save_data : list of str
                    List of names of saved buffers ( by default only 'mass')

        """
        # add_data = { False:[] , True:['quanta','quantization_norm']}[self.is_quantized]
        add_data = { False:[] , True:['quanta',]}[self.is_quantized]

        for a in saved_data+add_data:
            o = getattr(self, a)
            o[()] = hdf_file[fieldname][a][()]
        self.update_biomass_fields()
