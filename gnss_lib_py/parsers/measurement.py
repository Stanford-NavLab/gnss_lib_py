"""Base class for moving values between different modules/functions

"""

__authors__ = "Ashwin Kanhere, D. Knowles"
__date__ = "03 Nov 2021"

import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Measurement(ABC):
    """gnss_lib_py specific class for handling measurements.
    Uses numpy for speed combined with pandas like intuitive indexing

    Attributes
    ----------
    arr_dtype : numpy.dtype
        Type of values stored in data array
    array : np.ndarray
        Array containing measurements, dimension M x N
    map : Dict
        Map of the form {pandas column name : array row number }
    str_map : Dict
        Map of the form {pandas column name : {array value : string}}.
        Map is of the form {pandas column name : {}} for non string rows.
    """
    def __init__(self, csv_path=None, pandas_df=None, numpy_array=None):

        if csv_path is not None:
            self.from_csv_path(csv_path)
        elif pandas_df is not None:
            self.from_pandas_df(pandas_df)
        elif numpy_array is not None:
            self.from_numpy_array(numpy_array)
        else:
            self.build_measurement()

        self.postprocess()

    def postprocess(self):
        """Postprocess loaded measurements. Optional in subclass
        """
        pass

    def build_measurement(self):
        """Build attributes for Measurements.

        """
        self.arr_dtype = np.float64
        self.array = np.zeros(0, dtype=self.arr_dtype)

    def from_csv_path(self, csv_path):
        """Build attributes of Measurement using csv file.

        Parameters
        ----------
        csv_path : string
            Path to csv file containing data

        """
        if not isinstance(csv_path, str):
            raise TypeError("csv_path must be string")
        if not os.path.exists(csv_path):
            raise OSError("file not found")

        self.build_measurement()

        pandas_df = pd.read_csv(csv_path)
        self.from_pandas_df(pandas_df)

    def from_pandas_df(self, pandas_df):
        """Build attributes of Measurement using pd.DataFrame.

        Parameters
        ----------
        pandas_df : pd.DataFrame of measurements
        """

        if not isinstance(pandas_df, pd.DataFrame):
            raise TypeError("pandas_df must be pd.DataFrame")

        self.build_measurement()

        num_times = len(pandas_df)
        self.array = np.zeros([0, num_times], dtype= self.arr_dtype)
        # Using an empty array to conserve space and not maintain huge duplicates
        self.map = {col_name: idx for idx, col_name in enumerate(pandas_df.columns)}
        str_bool = {col_name: self.check_string(pandas_df[col_name]) \
                    for col_name in self.map.keys()}
        # Assuming that the data type of all objects in a single series is the same
        self.str_map = {}
        #TODO: See if we can avoid this loop to speed things up
        for key in self.map.keys():
            # indexing by key to maintain same order as eventual map
            val = str_bool[key]
            values = pandas_df.loc[:, key]
            new_values = np.copy(values)
            if val:
                string_vals = np.unique(pandas_df.loc[:, key])
                val_dict = dict(enumerate(string_vals))
                self.str_map[key] = val_dict

                for str_key, str_val in val_dict.items():
                    new_values[values==str_val] = str_key
                # Copy set to false to prevent memory overflows
                new_values = new_values.astype(self.arr_dtype, copy=False)
            else:
                self.str_map[key] = {}
            new_values = np.reshape(new_values, [1, -1])
            self.array = np.vstack((self.array, new_values))
            #TODO: Use __setitem__ here for self consistency
            #TODO: Rebuild map after all additions/deletions?

    def from_numpy_array(self, numpy_array):
        """Build attributes of Measurement using np.ndarray.

        Parameters
        ----------
        numpy_array : np.ndarray
            Numpy array containing data

        """

        if not isinstance(numpy_array, np.ndarray):
            raise TypeError("numpy_array must be np.ndarray")

        self.build_measurement()

    def pandas_df(self):
        """Return pandas DataFrame equivalent to class

        Returns
        -------
        df : pd.DataFrame
            DataFrame with measurements, including strings as strings
        """
        df_list = []
        for key, value in self.str_map.items():
            if value:
                vect_val = self.get_strings(key)
            else:
                vect_val = self.array[self.map[key], :]
            df_val = pd.DataFrame(vect_val, columns=[key])
            df_list.append(df_val)
        dframe = pd.concat(df_list, axis=1)
        return dframe

    def get_strings(self, key):
        """Return list of strings for given key

        Parameters
        ----------
        key : string for column name required as string

        Returns
        -------
        values_str : np.ndarray
            1D array with string entries corresponding to dataset
        """
        values_int = self.array[self.map[key],:].astype(int)
        values_str = values_int.astype(object, copy=True)
        # True by default but making explicit for clarity
        for str_key, str_val in self.str_map[key].items():
            values_str[values_int==str_key] = str_val
        return values_str

    def save_csv(self, output_path):
        """Save measurements as csv

        Parameters
        ----------
        output_path : string
            Path where csv should be saved
        """
        pd_df = self.pandas_df()
        pd_df.to_csv(output_path)

    def __getitem__(self, key_idx):
        """Return item indexed from class

        Parameters
        ----------
        key_idx : tuple
            tuple of form (row_name, idx). Row name can be string,
            list or 'all' for all rows

        Returns
        -------
        arr_slice : np.ndarray
            Array of measurements containing row names and time indexed
            columns
        """
        rows = []
        cols = key_idx[1]
        if isinstance(key_idx[0], slice):
            arr_slice = self.array[key_idx[0], cols]
        else:
            if not isinstance(key_idx[0],list):
                row_key = [key_idx[0]]
            else:
                row_key = key_idx[0]
            for key in row_key:
                rows.append(self.map[key])
            arr_slice = self.array[rows, cols]
        return arr_slice

    def __setitem__(self, key, newvalue):
        """Add/update rows

        Parameters
        ----------
        key : string
            Name of row to add/update

        newvalue : np.ndarray/list
            List or array of values to be added to measurements
        """
        #TODO: Currently breaks if you pass strings as np.ndarray
        if key=='all':
            # Check that number of rows is consistent
            #NOTE: Might not be needed if dictionary updated here
            assert np.shape(newvalue)[0] == self.shape[0], \
                "Inconsistent rows between old and new array"
            self.array = newvalue
        elif key in self.map.keys():
            if not isinstance(self[key, 0], type(newvalue[0])):
                raise TypeError("Type inconsistency in __setitem__")
            self.array[self.map[key], :] = newvalue
        else:
            values = newvalue
            self.map[key] = np.shape(self.array)[0]
            if isinstance(newvalue[0], str):
                assert isinstance(newvalue, np.ndarray), \
                    "Only numpy arrays supported with strings"
                #TODO: Replace this and in __init__ with private method?
                string_vals = np.unique(newvalue)
                val_dict = dict(enumerate(string_vals))
                self.str_map[key] = val_dict
                added_vals = len(string_vals)*np.ones(np.shape(newvalue), dtype=self.arr_dtype)
                # Set unassigned value to int not accessed by string map
                for str_key, str_val in val_dict.items():
                    added_vals[values==str_val] = str_key
                # Copy set to false to prevent memory overflows
                newvalue = np.round(added_vals.astype(self.arr_dtype, copy=False))
            else:
                self.str_map[key] = {}
            self.array = np.vstack((self.array, \
                        np.reshape(newvalue, [1, -1])))

    def __iter__(self):
        self.curr_col = 0
        self.num_cols = np.shape(self.array)[1]
        return self

    def __next__(self):
        if self.curr_col < self.num_cols:
            #TODO: Replace 'all' with slice for all rows
            x_curr = self['all', self.curr_col]
            self.curr_col += 1
            return x_curr
        else:
            raise StopIteration

    def __len__(self):
        """Return length of class

        Returns
        -------
        length : int
            Number of time steps in measurement
        """
        length = np.shape(self.array)[1]
        return length

    @property
    def shape(self):
        """Return shape of class

        Returns
        -------
        shp : tuple
            (M, N), M is number of rows and N number of time steps
        """
        shp = np.shape(self.array)
        return shp

    @property
    def rows(self):
        """Return all row names in instance as a list

        Returns
        -------
        rows : list
            List of row names in measurements
        """
        rows = list(self.map.keys())
        return rows


    def check_string(self, series):
        """Check if pd.Series contains any string values.

        """
        return series.dtype == object
