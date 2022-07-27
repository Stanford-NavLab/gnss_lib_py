"""Base class for moving values between different modules/functions

"""

__authors__ = "Ashwin Kanhere, D. Knowles"
__date__ = "03 Nov 2021"

import os
from abc import ABC
import copy

import numpy as np
import pandas as pd


class NavData(ABC):
    """gnss_lib_py specific class for handling data.
    Uses numpy for speed combined with pandas like intuitive indexing

    Attributes
    ----------
    arr_dtype : numpy.dtype
        Type of values stored in data array
    array : np.ndarray
        Array containing data, dimension M x N
    map : Dict
        Map of the form {pandas column name : array row number }
    str_map : Dict
        Map of the form {pandas column name : {array value : string}}.
        Map is of the form {pandas column name : {}} for non string rows.
    """
    def __init__(self, csv_path=None, pandas_df=None, numpy_array=None,
                 header="infer"):
        #For a Pythonic implementation, including all attributes as None in the beginning
        self.arr_dtype = np.float32 # default value
        self.array = None
        self.map = {}
        self.str_map = {}

        # Attributes for looping over all columns

        self.curr_col = 0
        self.num_cols = 0

        if csv_path is not None:
            self.from_csv_path(csv_path, header)
        elif pandas_df is not None:
            self.from_pandas_df(pandas_df)
        elif numpy_array is not None:
            self.from_numpy_array(numpy_array)
        else:
            self.build_navdata()

        self.postprocess()

    def postprocess(self):
        """Postprocess loaded data. Optional in subclass
        """

    def build_navdata(self):
        """Build attributes for NavDatas.

        """
        self.array = np.zeros((0,0), dtype=self.arr_dtype)

    def from_csv_path(self, csv_path, header="infer"):
        """Build attributes of NavData using csv file.

        Parameters
        ----------
        csv_path : string
            Path to csv file containing data
        header : string, int, or None
            "infer" uses the first row as column names, setting to
            None will add int names for the columns.

        """
        if not isinstance(csv_path, str):
            raise TypeError("csv_path must be string")
        if not os.path.exists(csv_path):
            raise OSError("file not found")

        self.build_navdata()

        pandas_df = pd.read_csv(csv_path, header=header)
        self.from_pandas_df(pandas_df)

    def from_pandas_df(self, pandas_df):
        """Build attributes of NavData using pd.DataFrame.

        Parameters
        ----------
        pandas_df : pd.DataFrame of data
        """

        if not isinstance(pandas_df, pd.DataFrame):
            raise TypeError("pandas_df must be pd.DataFrame")

        if pandas_df.columns.dtype != object:
            # default headers are Int64 type, but for the NavData
            # class they need to be strings
            pandas_df.rename(str, axis="columns", inplace=True)

        self.build_navdata()

        for _, col_name in enumerate(pandas_df.columns):
            newvalue = pandas_df[col_name].to_numpy()
            self.__setitem__(col_name, newvalue)

    def from_numpy_array(self, numpy_array):
        """Build attributes of NavData using np.ndarray.

        Parameters
        ----------
        numpy_array : np.ndarray
            Numpy array containing data

        """

        if not isinstance(numpy_array, np.ndarray):
            raise TypeError("numpy_array must be np.ndarray")

        self.build_navdata()

        for row_num in range(numpy_array.shape[0]):
            self.__setitem__(str(row_num), numpy_array[row_num,:])

    def pandas_df(self):
        """Return pandas DataFrame equivalent to class

        Returns
        -------
        df : pd.DataFrame
            DataFrame with data, including strings as strings
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
        """Save data as csv

        Parameters
        ----------
        output_path : string
            Path where csv should be saved
        """
        pd_df = self.pandas_df()
        pd_df.to_csv(output_path)

    def _parse_key_idx(self, key_idx):
        """Break down input queries to relevant row and column indices

        Parameters
        ----------
        key_idx : str/list/tuple/slice/int
            Query for array items

        Returns
        -------
        rows : slice/list
            Rows to extract from the array
        cols : slice/list
            Columns to extract from the array
        """
        if isinstance(key_idx, str):
            rows = [self.map[key_idx]]
            cols = slice(None, None)
        elif isinstance(key_idx, list) and isinstance(key_idx[0], str):
            rows = [self.map[k] for k in key_idx]
            cols = slice(None, None)
        elif isinstance(key_idx, slice):
            rows = key_idx
            cols = slice(None, None)
        #TODO: Add int testing for just row number
        elif isinstance(key_idx, int):
            rows = [key_idx]
            cols = slice(None, None)
        else:
            if isinstance(key_idx[1], int):
                cols = [key_idx[1]]
            else:
                cols = key_idx[1]

            rows = []
            if isinstance(key_idx[0], slice):
                rows = key_idx[0]
            elif isinstance(key_idx[0], int):
                rows = [key_idx[0]]
            else:
                if not isinstance(key_idx[0],list):
                    row_key = [key_idx[0]]
                else:
                    row_key = key_idx[0]
                for key in row_key:
                    rows.append(self.map[key])
        #TODO: Add out of bounds error handling
        return rows, cols

    def _get_str_rows(self, rows):
        """Checks which input rows contain string elements

        Parameters
        ----------
        rows : slice/list
            Rows to check for string elements

        Returns
        -------
        row_list : list
            Input rows, strictly in list format
        row_str : list
            List of boolean values indicating which rows contain strings
        """
        str_bool = self.str_bool
        if isinstance(rows, slice):
            slice_idx = rows.indices(self.shape[0])
            row_list = np.arange(slice_idx[0], slice_idx[1], slice_idx[2])
            row_str = [str_bool[row] for row in row_list]
        else:
            row_list = list(rows)
            row_str = [str_bool[row] for row in rows]
        return row_list, row_str

    def __getitem__(self, key_idx):
        """Return item indexed from class

        Parameters
        ----------
        key_idx : str/list/tuple/slice/int
            Query for array items

        Returns
        -------
        arr_slice : np.ndarray
            Array of data containing row names and time indexed
            columns
        """
        rows, cols = self._parse_key_idx(key_idx)
        row_list, row_str = self._get_str_rows(rows)
        assert np.all(row_str) or np.all(np.logical_not(row_str)), \
                "Cannot assign/return combination of strings and numbers"
        if np.all(row_str):
            # Return sliced strings
            # arr_slice  = []
            arr_slice = np.atleast_2d(np.empty_like(self.array[rows, cols], dtype=object))
            for row_num, row in enumerate(row_list):
                str_arr = self.get_strings(self.inv_map[row])
                arr_slice[row_num, :] = str_arr[cols]
                # arr_slice.append(str_arr[ cols])
        else:
            arr_slice = self.array[rows, cols]
        return arr_slice

    def __setitem__(self, key_idx, newvalue):
        """Add/update rows.

        Parameters
        ----------
        key_idx : str/list/tuple/slice/int
            Query for array items to set

        newvalue : np.ndarray/list/int
            Values to be added to self.array attribute
        """
        #TODO: Fix error when assigning strings with 2D arrays
        if isinstance(key_idx, int) and len(self.map)<=key_idx:
            raise KeyError('Row indices must be strings when assigning new values')
        if isinstance(key_idx, slice) and len(self.map)==0:
            raise KeyError('Row indices must be strings when assigning new values')
        if isinstance(key_idx, str) and key_idx not in self.map.keys():
            #Creating an entire new row
            if isinstance(newvalue, np.ndarray) and newvalue.dtype==object:
                # Adding string values
                self.fillna(newvalue)
                new_str_vals = len(np.unique(newvalue))*np.ones(np.shape(newvalue), dtype=self.arr_dtype)
                new_str_vals = self._str_2_val(new_str_vals, newvalue, key_idx)
                if self.array.shape == (0,0):
                    # if empty array, start from scratch
                    self.array = np.reshape(new_str_vals, [1, -1])
                else:
                    # if array is not empty, add to it
                    self.array = np.vstack((self.array, np.reshape(new_str_vals, [1, -1])))
                self.map[key_idx] = self.shape[0]-1
            else:
                if not isinstance(newvalue, int):
                    assert not isinstance(np.asarray(newvalue)[0], str), \
                            "Cannot set a row with list of strings, please use np.ndarray with dtype=object"
                # Adding numeric values
                self.str_map[key_idx] = {}
                if self.array.shape == (0,0):
                    # if empty array, start from scratch
                    self.array = np.reshape(newvalue, (1,-1))
                    # have to explicitly convert to float in case
                    # numbers were interpretted as integers
                    self.array = self.array.astype(self.arr_dtype)
                else:
                    # if array is not empty, add to it
                    self.array = np.vstack((self.array, np.empty([1, len(self)])))
                    self.array[-1, :] = np.reshape(newvalue, -1)
                self.map[key_idx] = self.shape[0]-1
        else:
            # Updating existing rows or columns
            rows, cols = self._parse_key_idx(key_idx)
            row_list, row_str = self._get_str_rows(rows)
            assert np.all(row_str) or np.all(np.logical_not(row_str)), \
                "Cannot assign/return combination of strings and numbers"
            if np.all(row_str):
                assert isinstance(newvalue, np.ndarray) and newvalue.dtype==object, \
                        "String assignment only supported for ndarray of type object"
                inv_map = self.inv_map
                newvalue = np.reshape(newvalue, [-1, newvalue.shape[0]])
                new_str_vals = np.ones_like(newvalue, dtype=self.arr_dtype)
                for row_num, row in enumerate(row_list):
                    # print('Assigning values to ', inv_map[row])
                    key = inv_map[row]
                    newvalue_row = newvalue[row_num , :]
                    new_str_vals_row = new_str_vals[row_num, :]
                    new_str_vals[row_num, :] = self._str_2_val(new_str_vals_row, newvalue_row, key)
                self.array[rows, cols] = new_str_vals
            else:
                if not isinstance(newvalue, int):
                    assert not isinstance(np.asarray(newvalue)[0], str), \
                            "Please use dtype=object for string assignments"
                self.array[rows, cols] = newvalue

    def _str_2_val(self, new_str_vals, newvalue, key):
        """Convert string valued arrays to values for storing in array

        Parameters
        ----------
        new_str_vals : np.ndarray
            Array of dtype=self.arr_dtype where numeric values are to be stored
        newvalue : np.ndarray
            Array of dtype=object, containing string values that are to be converted
        key : string
            Key indicating row where string to numeric conversion is required
        """
        if key in self.map.keys():
            # Key already exists, update existing string value dictionary
            inv_str_map = {v: k for k, v in self.str_map[key].items()}
            string_vals = np.unique(newvalue)
            str_map_dict = self.str_map[key]
            total_str = len(self.str_map[key])
            for str_val in string_vals:
                if str_val not in inv_str_map.keys():
                    str_map_dict[total_str] = str_val
                    new_str_vals[newvalue==str_val] = total_str
                    total_str += 1
                else:
                    new_str_vals[newvalue==str_val] = inv_str_map[str_val]
            self.str_map[key] = str_map_dict
        else:
            string_vals = np.unique(newvalue)
            str_dict = dict(enumerate(string_vals))
            self.str_map[key] = str_dict
            new_str_vals = len(string_vals)*np.ones(np.shape(newvalue), dtype=self.arr_dtype)
            # Set unassigned value to int not accessed by string map
            for str_key, str_val in str_dict.items():
                new_str_vals[newvalue==str_val] = str_key
            # Copy set to false to prevent memory overflows
            new_str_vals = np.round(new_str_vals.astype(self.arr_dtype, copy=False))
        return new_str_vals

    def add(self, csv_path=None, pandas_df=None, numpy_array=None):
        """Add new timesteps to existing array

        Parameters
        ----------
        csv_path : string
            Path to csv file containing data to add
        pandas_df : pd.DataFrame
            DataFrame containing data to add
        numpy_array : np.ndarray
            Array containing only numeric data to add
        """
        old_len = len(self)
        new_data_cols = slice(old_len, None)
        if numpy_array is not None:
            self.array = np.hstack((self.array, np.empty_like(numpy_array, dtype=self.arr_dtype)))
            self[:, new_data_cols] = numpy_array
        if csv_path is not None:
            pandas_df = pd.read_csv(csv_path)
        if pandas_df is not None:
            #TODO: Case handlign for when column name in dataframe is different?
            self.array = np.hstack((self.array, np.empty(pandas_df.shape).T))
            for col in pandas_df.columns:
                self[col, new_data_cols] = np.asarray(pandas_df[col].values)

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
            Number of time steps in NavData
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
            List of row names in NavData
        """
        rows = list(self.map.keys())
        return rows


    @property
    def str_bool(self):
        """Dictionary of index : if data entry is string.

        Row has string values if the string map is nonempty for a
        given row.

        Returns
        -------
        str_bool : Dict
            Dictionary of whether data at row number key is string or not
        """
        str_bool = {self.map[k]: bool(len(self.str_map[k])) for k in self.str_map.keys()}
        return str_bool

    @property
    def inv_map(self):
        """Inverse dictionary map for label and row_number map

        Returns
        -------
        inv_map: Dict
            Dictionary of row_number : label
        """
        inv_map = {v: k for k, v in self.map.items()}
        return inv_map

    def fillna(self, array):
        """Fills nan values in an array of strings.

        You have to do a string comparison, so we first have to create
        the string equivalent of the NaN to compare against.

        array : np.ndarray
            np.ndarray of strings

        """
        nan_str = np.array([np.nan]).astype(str)[0]
        array[np.where(array.astype(str)==nan_str)] = ""

    def rename(self, mapper):
        """Rename rows of NavData class.

        Column names must be strings.

        mapper : dict
            Pairs of {"old_name" : "new_name"} for each column that
            you want to be renamed.

        """

        for key, value in mapper.items():
            if not isinstance(value, str):
                raise TypeError("Column names must be strings")
            if key not in self.map.keys():
                raise KeyError("'" + str(key) + "' key doesn't exist in NavData class")

            self.map[value] = self.map.pop(key)
            self.str_map[value] = self.str_map.pop(key)

    def copy(self, rows=None, cols=None):
        """Return copy of NavData keeping specified rows and columns

        Parameters
        ----------
        rows : None/list/np.ndarray
            Strings or integers indicating rows to keep in copy
        cols : None/list/np.ndarray
            Integers indicating columns to keep in copy

        Returns
        -------
        new_navdata : gnss_lib_py.parsers.navdata.NavData
            Copy of original NavData with desired rows and columns
        """
        new_navdata = NavData()
        inv_map = self.inv_map
        if rows is None:
            # row_indices = slice(None, None).indices(len(self.rows))
            # rows = np.arange(row_indices[0], row_indices[1], row_indices[2])
            rows = self.rows
        if cols is None:
            col_indices = slice(None, None).indices(len(self))
            cols = np.arange(col_indices[0], col_indices[1], col_indices[2])
        for row_idx in rows:
            new_row = copy.deepcopy(self[row_idx, cols])
            if isinstance(row_idx, int):
                key = inv_map[row_idx]
            else:
                key = row_idx
            new_navdata[key] = new_row
        return new_navdata

    def remove(self, rows=None, cols=None):
        """Reset NavData to remove specified rows and columns

        Parameters
        ----------
        rows : None/list/np.ndarray
            Rows to remove from NavData
        cols : None/list/np.ndarray
            Columns to remove from NavData

        Returns
        -------
        new_navdata : gnss_lib_py.parsers.navdata.NavData
            NavData instance after removing specified rows and columns

        Notes
        -----
        This method returns the NavData with removed rows and columns,
        but does not reset the current instance in place.
        """
        if cols is None:
            cols = []
        if rows is None:
            rows = []
        new_navdata = NavData()
        inv_map = self.inv_map
        if len(rows) != 0 and isinstance(rows[0], int):
            rows = [inv_map[row_idx] for row_idx in rows]
        keep_rows = [row for row in self.rows if row not in rows]
        keep_cols = [col for col in range(len(self)) if col not in cols]
        for row_idx in keep_rows:
            new_row = self[row_idx, keep_cols]
            key = row_idx
            new_navdata[key] = new_row
        return new_navdata