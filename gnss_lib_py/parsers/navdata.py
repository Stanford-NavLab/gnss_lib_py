"""Base class for moving values between different modules/functions

"""

__authors__ = "Ashwin Kanhere, D. Knowles"
__date__ = "03 Nov 2021"

import os
import re
import copy

import numpy as np
import pandas as pd


class NavData():
    """gnss_lib_py specific class for handling data.

    Uses numpy for speed combined with pandas like intuitive indexing.

    Can either be initialized empty, with a csv file by setting
    ``csv_path``, a Pandas DataFrame by setting ``pandas_df`` or by a
    Numpy array by setting ``numpy_array``.

    Parameters
    ----------
    csv_path : string
        Path to csv file containing data
    pandas_df : pd.DataFrame
        Data used to initialize NavData instance.
    numpy_array : np.ndarray
        Numpy array containing data used to initialize NavData
        instance.
    **kwargs : args
        Additional arguments (e.g. ``sep`` or ``header``) passed into
        ``pd.read_csv`` if csv_path is not None.

    Attributes
    ----------
    arr_dtype : numpy.dtype
        Type of values stored in data array
    orig_dtypes : pandas.core.series.Series
        Type of each original column if reading from a csv or Pandas
        dataframe.
    array : np.ndarray
        Array containing data, dimension M x N
    map : Dict
        Map of the form {pandas column name : array row number }
    str_map : Dict
        Map of the form {pandas column name : {array value : string}}.
        Map is of the form {pandas column name : {}} for non string rows.
    num_cols : int
        Number of columns in array containing data, set to 0 by default
        for empty NavData
    curr_cols : int
        Current number of column for iterator, set to 0 by default

    """
    def __init__(self, csv_path=None, pandas_df=None, numpy_array=None,
                 **kwargs):
        # For a Pythonic implementation,
        # including all attributes as None in the beginning
        self.arr_dtype = np.float64 # default value
        self.orig_dtypes = {}       # original dtypes
        self.array = None
        self.map = {}
        self.str_map = {}

        # Attributes for looping over all columns

        self.curr_col = 0
        self.num_cols = 0

        if csv_path is not None:
            self.from_csv_path(csv_path, **kwargs)
        elif pandas_df is not None:
            self.from_pandas_df(pandas_df)
        elif numpy_array is not None:
            self.from_numpy_array(numpy_array)
        else:
            self._build_navdata()

        self.rename(self._row_map(), inplace=True)

        self.postprocess()

    def postprocess(self):
        """Postprocess loaded data. Optional in subclass
        """

    def from_csv_path(self, csv_path, **kwargs):
        """Build attributes of NavData using csv file.

        Parameters
        ----------
        csv_path : string
            Path to csv file containing data
        header : string, int, or None
            "infer" uses the first row as column names, setting to
            None will add int names for the columns.
        sep : char
            Delimiter to use when reading in csv file.

        """
        if not isinstance(csv_path, str):
            raise TypeError("csv_path must be string")
        if not os.path.exists(csv_path):
            raise FileNotFoundError("file not found")

        self._build_navdata()

        pandas_df = pd.read_csv(csv_path, **kwargs)
        self.from_pandas_df(pandas_df)

    def from_pandas_df(self, pandas_df):
        """Build attributes of NavData using pd.DataFrame.

        Parameters
        ----------
        pandas_df : pd.DataFrame
            Data used to initialize NavData instance.
        """

        if not isinstance(pandas_df, pd.DataFrame):
            raise TypeError("pandas_df must be pd.DataFrame")

        dtypes = dict(pandas_df.dtypes)
        for row, dtype in dtypes.items():
            if np.issubdtype(dtype,np.integer):
                dtype = np.int64
            self.orig_dtypes[row] = dtype

        if pandas_df.columns.dtype != object:
            # default headers are Int64 type, but for the NavData
            # class they need to be strings
            pandas_df.rename(str, axis="columns", inplace=True)

        self._build_navdata()

        for _, col_name in enumerate(pandas_df.columns):
            new_value = pandas_df[col_name].to_numpy()
            self[col_name] = new_value

    def from_numpy_array(self, numpy_array):
        """Build attributes of NavData using np.ndarray.

        Parameters
        ----------
        numpy_array : np.ndarray
            Numpy array containing data used to initialize NavData
            instance.

        """

        if not isinstance(numpy_array, np.ndarray):
            raise TypeError("numpy_array must be np.ndarray")

        self._build_navdata()

        numpy_array = np.atleast_2d(numpy_array)
        for row_num in range(numpy_array.shape[0]):
            self[str(row_num)] = numpy_array[row_num,:]

    def concat(self, navdata=None, axis=1, inplace=False):
        """Concatenates second NavData instance by row or column.

        Concatenates a second NavData instance to the existing NavData
        instance by either row or column.

        Each type of data is included in a row, so adding new rows with
        ``axis=0``, means adding new types of data. Concat requires that
        the new NavData matches the length of the existing NavData. Row
        concatenation assumes the same ordering across both NavData
        instances (e.g. sorted by timestamp) and does not perform any
        matching/sorting itself.

        You can also concatenate new columns ``axis=1``. If the row
        names of the new NavData instance don't match the row names of
        the existing NavData instance, the mismatched values will be
        filled with np.nan.

        Parameters
        ----------
        navdata : gnss_lib_py.parsers.navdata.NavData
            Navdata instance to concatenate.
        axis : int
            Either add new rows (type) of data ``axis=0`` or new columns
            (e.g. timesteps) of data ``axis=1``.
        inplace : bool
            If False, will return new concatenated NavData instance.
            If True, will concatenate data to the current NavData
            instance.

        Returns
        -------
        new_navdata : gnss_lib_py.parsers.navdata.NavData or None
            If inplace is False, returns NavData instance after
            concatenating specified data. If inplace is True, returns
            None.

        """

        if not isinstance(navdata,NavData):
            raise TypeError("concat input data must be a NavData instance.")

        if axis == 0: # concatenate new rows
            if len(self) != len(navdata):
                raise RuntimeError("concat input data must be same " \
                                 + "length to concatenate new rows.")
            if not inplace:
                new_navdata = self.copy()
            for row in navdata.rows:
                new_row_name = row
                suffix = None
                while new_row_name in self.rows:
                    if suffix is None:
                        suffix = 0
                    else:
                        suffix += 1
                    new_row_name = row + "_" + str(suffix)
                new_row = navdata[row].astype(navdata.orig_dtypes[row])
                if inplace:
                    self[new_row_name] = new_row
                else:
                    new_navdata[new_row_name] = new_row

        elif axis == 1: # concatenate new columns
            new_navdata = NavData()
            # get unique list of row names
            combined_rows = self.rows  + [row for row in navdata.rows
                                          if row not in self.rows]

            for row in combined_rows:
                combined_row = np.array([])
                # combine data from existing and new instance
                for data in [self, navdata]:
                    if row in data.rows:
                        new_row = np.atleast_1d(data[row])
                    elif len(data) == 0:
                        continue
                    else:
                        # add np.nan for missing values
                        new_row = np.empty((len(data),))
                        new_row.fill(np.nan)
                    combined_row = np.concatenate((combined_row,
                                                   new_row))
                new_navdata[row] = combined_row

            if len(self) > 0 and len(navdata) > 0:
                new_navdata.orig_dtypes = navdata.orig_dtypes.copy()
                new_navdata.orig_dtypes.update(self.orig_dtypes)
            elif len(self) > 0:
                new_navdata.orig_dtypes = self.orig_dtypes.copy()
            elif len(navdata) > 0:
                new_navdata.orig_dtypes = navdata.orig_dtypes.copy()

            if inplace:
                self.array = new_navdata.array
                self.map = new_navdata.map
                self.str_map = new_navdata.str_map
                self.orig_dtypes = new_navdata.orig_dtypes.copy()

        if inplace:
            return None
        return new_navdata

    def where(self, key_idx, value, condition="eq"):
        """Return NavData where conditions are met for the given row.

        For string rows, only the "eq" and "neq" conditions are valid.
        The "value" argument can contain either a string, np.nan or an
        array-like object of strings. If an array-like object of strings
        is passed in then np.isin() is used to check the condition
        meaning that the returned subset will contain one of the values
        in the "value" array-like object for the "eq" condition or none
        of the values in the "value" array-like object for the "neq"
        condition.

        For non-string rows, all valid conditions are listed in the
        "condition" argument description. The "value" argument can either
        contain a numeric or an array-like object of numerics for both
        the "eq" and "neq" conditions.
        If an array-like object is passed then the returned subset will
        contain one of the values in the "value" array-like object for
        the "eq" condition or none of the values in the "value"
        array-like object for the "neq" condition.
        For the "between" condition, the two limit values must be passed
        into the "value" argument as an array-like object.

        Parameters
        ----------
        key_idx : string/int
            Key or index of the row in which conditions will be checked
        value : float/int/str/array-like
            Value that the row is checked against, array-like object
            possible for "eq", "neq", or "between" conditions.
        condition : string
            Condition type (greater than ("greater")/ less than ("lesser")/
            equal to ("eq")/ greater than or equal to ("geq")/
            lesser than or equal to ("leq") / in between ("between")
            inclusive of the provided limits / not equal to ("neq"))

        Returns
        -------
        new_navdata : gnss_lib_py.parsers.navdata.NavData
            NavData with columns where given condition is satisfied
            for specified row
        """
        new_cols = self.argwhere(key_idx, value, condition)
        if new_cols.size == 0:
            return self.remove(cols=list(range(len(self))))
        new_navdata = self.copy(cols=new_cols)
        return new_navdata

    def argwhere(self, key_idx, value, condition="eq"):
        """Return columns where conditions are met for the given row.

        For string rows, only the "eq" and "neq" conditions are valid.
        The "value" argument can contain either a string, np.nan or an
        array-like object of strings. If an array-like object of strings
        is passed in then np.isin() is used to check the condition
        meaning that the returned subset will contain one of the values
        in the "value" array-like object for the "eq" condition or none
        of the values in the "value" array-like object for the "neq"
        condition.

        For non-string rows, all valid conditions are listed in the
        "condition" argument description. The "value" argument can either
        contain a numeric or an array-like object of numerics for both
        the "eq" and "neq" conditions.
        If an array-like object is passed then the returned subset will
        contain one of the values in the "value" array-like object for
        the "eq" condition or none of the values in the "value"
        array-like object for the "neq" condition.
        For the "between" condition, the two limit values must be passed
        into the "value" argument as an array-like object.

        Parameters
        ----------
        key_idx : string/int
            Key or index of the row in which conditions will be checked
        value : float/int/str/array-like
            Value that the row is checked against, array-like object
            possible for "eq", "neq", or "between" conditions.
        condition : string
            Condition type (greater than ("greater")/ less than ("lesser")/
            equal to ("eq")/ greater than or equal to ("geq")/
            lesser than or equal to ("leq") / in between ("between")
            inclusive of the provided limits / not equal to ("neq"))

        Returns
        -------
        new_cols : list
            Columns in NavData where given condition is satisfied
            for specified row
        """
        rows, _ = self._parse_key_idx(key_idx)
        row_list, row_str = self._get_str_rows(rows)
        if len(row_list)>1:
            error_msg = "where does not currently support multiple rows"
            raise NotImplementedError(error_msg)
        row = row_list[0]
        row_str = row_str[0]
        new_cols = np.array([])
        if row_str:
            # Values in row are strings
            if condition not in ("eq","neq"):
                raise ValueError("Unsupported where condition for strings")
            if isinstance(value,str):
                str_check = [str(value)]
            elif isinstance(value,(np.ndarray,list,tuple,set)):
                str_check = [str(v) for v in value]
            elif np.isnan(value):
                str_check = [str(np.nan)]
            else:
                raise ValueError("Value must be string or array-like " \
                               + "for string condition checks")
            # Extract columns where condition holds true and return new NavData
            if condition == "eq":
                new_cols = np.argwhere(np.isin(self[row, :],str_check))
            else:
                # condition == "neq"
                new_cols = np.argwhere(~np.isin(self[row, :],str_check))

        else:
            # Values in row are numerical
            # Find columns where value can be found and return new NavData
            if condition=="eq":
                if isinstance(value,(np.ndarray,list,tuple,set)):
                    # use numpy's isin() condition if list of values
                    new_cols = np.argwhere(np.isin(self.array[row, :],
                                           value))
                elif not isinstance(value,str) and np.isnan(value):
                    # check isinstance b/c np.isnan can't handle strings
                    new_cols = np.argwhere(np.isnan(self.array[row, :]))
                else:
                    new_cols = np.argwhere(self.array[row, :]==value)
            elif condition=="neq":
                if isinstance(value,(np.ndarray,list,tuple,set)):
                    # use numpy's isin() condition if list of values
                    new_cols = np.argwhere(~np.isin(self.array[row, :],
                                           value))
                elif not isinstance(value,str) and np.isnan(value):
                    # check isinstance b/c np.isnan can't handle strings
                    new_cols = np.argwhere(~np.isnan(self.array[row, :]))
                else:
                    new_cols = np.argwhere(self.array[row, :]!=value)
            elif condition == "leq":
                new_cols = np.argwhere(self.array[row, :]<=value)
            elif condition == "geq":
                new_cols = np.argwhere(self.array[row, :]>=value)
            elif condition == "greater":
                new_cols = np.argwhere(self.array[row, :]>value)
            elif condition == "lesser":
                new_cols = np.argwhere(self.array[row, :]<value)
            elif condition == "between":
                assert len(value)==2, "Please give both lower and upper bound for between"
                new_cols = np.argwhere(np.logical_and(self.array[row, :]>=value[0],
                                        self.array[row, :]<= value[1]))
            else:
                raise ValueError("Condition not implemented")
        new_cols = np.squeeze(new_cols)
        return new_cols

    def sort(self, order=None, ind=None, ascending=True,
             inplace=False):
        """Sort values along given row or using given index

        Parameters
        ----------
        order : string/int
            Key or index of the row on which NavData will be sorted
        ind : list/np.ndarray
            Ordering of indices to be used for sorting
        ascending : bool
            If true, sorts "ascending", otherwise sorts "descending"
        inplace : bool
            If False, will return new NavData instance with rows
            sorted. If True, will sorted data rows in the
            current NavData instance.

        Returns
        -------
        new_navdata : gnss_lib_py.parsers.navdata.NavData or None
            If inplace is False, returns NavData instance after renaming
            specified rows. If inplace is True, returns
            None.

        """
        if ind is None:
            assert order is not None, \
            "Provide 'order' arg as row on which NavData is sorted"
            if ascending:
                ind = np.argsort(self[order])
            else:
                ind = np.argsort(-self[order])

        if not inplace:
            new_navdata = self.copy()   # create copy to return
        for row_idx in range(self.shape[0]):
            if inplace:
                self.array[row_idx,:] = self.array[row_idx,ind]
            else:
                new_navdata.array[row_idx,:] = new_navdata.array[row_idx,ind]

        if inplace:
            return None
        return new_navdata

    def loop_time(self, time_row, delta_t_decimals=2):
        """Generator object to loop over columns from same times.

        Parameters
        ----------
        time_row : string/int
            Key or index of the row in which times are stored.
        delta_t_decimals : int
            Decimal places after which times are considered equal.

        Yields
        ------
        timestamp : float
            Current timestamp.
        delta_t : float
            Difference between current time and previous time.
        new_navdata : gnss_lib_py.parsers.navdata.NavData
            NavData with same time, up to given decimal tolerance.

        """

        times = self[time_row]
        times_unique = np.sort(np.unique(np.around(times,
                                         decimals=delta_t_decimals)))
        for time_idx, time in enumerate(times_unique):
            if time_idx==0:
                delta_t = 0
            else:
                delta_t = time-times_unique[time_idx-1]
            new_navdata = self.where(time_row, [time-10**(-delta_t_decimals),
                                                time+10**(-delta_t_decimals)],
                                                condition="between")
            if len(np.unique(new_navdata[time_row]))==1:
                frame_time = new_navdata[time_row, 0]
            else:
                frame_time = time
            yield frame_time, delta_t, new_navdata

    def is_str(self, row_name):
        """Check whether a row contained string values.

        Parameters
        ----------
        row_name : string
            Name of the row to check whether it contains string values.

        Returns
        -------
        contains_str : bool
            True if the row contains string values, False otherwise.

        """

        if row_name not in self.map:
            raise KeyError("'" + str(row_name) \
                           + "' key doesn't exist in NavData class")

        contains_str = self._row_idx_str_bool[self.map[row_name]]

        return contains_str

    def rename(self, mapper=None, inplace=False):
        """Rename rows of NavData class.

        Row names must be strings.

        Parameters
        ----------
        mapper : dict
            Pairs of {"old_name" : "new_name"} for each row to be
            renamed.
        inplace : bool
            If False, will return new NavData instance with rows
            renamed. If True, will rename data rows in the
            current NavData instance.

        Returns
        -------
        new_navdata : gnss_lib_py.parsers.navdata.NavData or None
            If inplace is False, returns NavData instance after renaming
            specified rows. If inplace is True, returns
            None.

        """

        if not isinstance(mapper, dict):
            raise TypeError("'mapper' must be dict")
        if not isinstance(inplace, bool):
            raise TypeError("'inplace' must be bool")
        for old_name in mapper:
            if old_name not in self.map:
                raise KeyError("'" + str(old_name) + "' row name " \
                             + "doesn't exist in NavData class")

        if not inplace:
            new_navdata = self.copy()   # create copy to return
        for old_name, new_name in mapper.items():
            if not isinstance(new_name, str):
                raise TypeError("New row names must be strings")
            if inplace:
                self.map[new_name] = self.map.pop(old_name)
                self.str_map[new_name] = self.str_map.pop(old_name)
                self.orig_dtypes[new_name] = self.orig_dtypes.pop(old_name)
            else:
                new_navdata.map[new_name] = new_navdata.map.pop(old_name)
                new_navdata.str_map[new_name] = new_navdata.str_map.pop(old_name)
                new_navdata.orig_dtypes[new_name] = new_navdata.orig_dtypes.pop(old_name)

        if inplace:
            return None
        return new_navdata

    def replace(self, mapper=None, rows=None, inplace=False):
        """Replace data within rows or row names of NavData class.

        Row names must be strings.

        Parameters
        ----------
        mapper : dict
            Pairs of {"old_name" : "new_name"} for each value to
            replace. Values are replaced for each row in "rows" if
            "rows" is specified or for all rows if "rows" is left
            defaulted to None.
        rows : dict, or array-like
            If a dictionary is passed, then rows will be renamed
            according to pairs of {"old_name" : "new_name"}.
            If mapper is not None, then an array-like input may be
            passed to indicate which rows of values should be remapped.
        inplace : bool
            If False, will return new NavData instance with data
            replaced. If True, will replace data in the current NavData
            instance.

        Returns
        -------
        new_navdata : gnss_lib_py.parsers.navdata.NavData or None
            If inplace is False, returns NavData instance after
            replacing specified data. If inplace is True, returns None.

        """

        if not isinstance(mapper, dict):
            raise TypeError("'mapper' must be dict")
        if isinstance(rows,str):
            rows = [rows]
        if not (type(rows) in (dict, list, np.ndarray, tuple, set) \
           or rows is None):
            raise TypeError("'rows' must be dict, array-like or None")
        if not isinstance(inplace, bool):
            raise TypeError("'inplace' must be bool")
        if rows is not None:
            for old_name in rows:
                if old_name not in self.map:
                    raise KeyError("'" + str(old_name) + "' row name " \
                                 + "doesn't exist in NavData class")
        if rows is not None:
            # convert to None if rows is emptry
            rows = None if len(rows)==0 else rows
        if not inplace:
            new_navdata = self.copy()   # create copy to return
        if mapper is not None and len(self) > 0:
            remap_rows = self.rows if rows is None else rows
            for row in remap_rows:
                new_row_values = list(self[row])
                for old_value, new_value in mapper.items():
                    new_row_values = [new_value if v == old_value else v for v in new_row_values]
                if inplace:
                    self[row] = np.array(new_row_values)
                else:
                    new_navdata[row] = np.array(new_row_values)

        if inplace:
            return None
        return new_navdata

    def copy(self, rows=None, cols=None):
        """Return copy of NavData keeping specified rows and columns

        If None is passed into either argument, all rows or cols
        respectively are returned.

        If no arguments are added .copy() returns a full copy of the
        entire NavData class.

        Parameters
        ----------
        rows : None/list/np.ndarray
            Strings or integers indicating rows to keep in copy.
            Defaults to None meaning all rows are copied.
        cols : None/list/np.ndarray
            Integers indicating columns to keep in copy. Defaults to
            None meaning all cols are copied.

        Returns
        -------
        new_navdata : gnss_lib_py.parsers.navdata.NavData
            Copy of original NavData with desired rows and columns
        """
        new_navdata = NavData()
        new_navdata.arr_dtype = self.arr_dtype
        inv_map = self.inv_map
        if rows is None:
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

        new_navdata.orig_dtypes = self.orig_dtypes.copy()

        return new_navdata

    def remove(self, rows=None, cols=None, inplace=False):
        """Reset NavData to remove specified rows and columns

        Parameters
        ----------
        rows : None/list/np.ndarray/tuple
            Rows to remove from NavData
        cols : None/list/np.ndarray/tuple
            Columns to remove from NavData
        inplace : bool
            If False, will return new NavData instance with specified
            rows and columns removed. If True, will remove rows and
            columns from the current NavData instance.

        Returns
        -------
        new_navdata : gnss_lib_py.parsers.navdata.NavData or None
            If inplace is False, returns NavData instance after removing
            specified rows and columns. If inplace is True, returns
            None.

        """
        if cols is None:
            cols = []
        if rows is None:
            rows = []
        if isinstance(rows,str):
            rows = [rows]
        new_navdata = NavData()
        if len(rows) != 0 and isinstance(rows[0], int):
            try:
                rows = [self.inv_map[row_idx] for row_idx in rows]
            except KeyError as exception:
                raise KeyError("row '" + str(exception) + "' is out " \
                             + "of bounds of data.") from Exception

        for row in rows:
            if row not in self.rows:
                raise KeyError("row '" + row + "' does not exist so " \
                             + "cannont be removed.")
        for col in cols:
            if col >= len(self):
                raise KeyError("column '" + str(col) + "' exceeds " \
                             + "NavData dimensions, so cannont be " \
                             + "removed.")

        if inplace: # remove rows/cols from current column

            # delete rows and columns from self.array
            del_row_idxs = [self.map[row] for row in rows]
            self.array = np.delete(self.array,del_row_idxs,axis=0)
            self.array = np.delete(self.array,cols,axis=1)

            # delete keys from self.map and self.str_map
            for row in rows:
                del self.map[row]
                del self.str_map[row]

            # reindex self.map
            self.map = {key : index for index, key in \
                enumerate(sorted(self.map, key=lambda k: self.map[k]))}

            return None

        # inplace = False; return new instance with rows/cols removed
        keep_rows = [row for row in self.rows if row not in rows]
        keep_cols = [col for col in range(len(self)) if col not in cols]
        for row_idx in keep_rows:
            new_row = self[row_idx, keep_cols]
            key = row_idx
            new_navdata[key] = new_row
        return new_navdata

    def interpolate(self, x_row, y_rows, inplace=False, *args):
        """Interpolate NaN values based on row data.

        Additional ``*args`` arguments are passed into the ``np.interp``
        function.

        Parameters
        ----------
        x_row : string
            Row name for x-coordinate of all values (e.g. gps_millis).
            Row must not contain any nan values.
        y_rows : list or string
            Row name(s) for y-coordinate which includes nan values that
            will be interpolated.
        inplace : bool
            If False, will return new NavData instance with nan values
            interpolated. If True, will interpolate nan values within
            the current NavData instance.

        Returns
        -------
        new_navdata : gnss_lib_py.parsers.navdata.NavData or None
            If inplace is False, returns NavData instance after removing
            specified rows and columns. If inplace is True, returns
            None.

        """

        if isinstance(y_rows,str):
            y_rows = [y_rows]
        if not isinstance(x_row, str):
            raise TypeError("'x_row' must be row name as a string.")
        if not isinstance(y_rows, list):
            raise TypeError("'y_rows' must be single or list of " \
                          + "row names as a string.")
        self.in_rows([x_row] + y_rows)

        if not inplace:
            new_navdata = self.copy()
        for y_row in y_rows:
            nan_idxs = self.argwhere(y_row,np.nan)
            if nan_idxs.size == 0:
                continue
            not_nan_idxs = self.argwhere(y_row,np.nan,"neq")
            x_vals = self[x_row,nan_idxs]
            xp_vals = self[x_row,not_nan_idxs]
            yp_vals = self[y_row,not_nan_idxs]

            if inplace:
                self[y_row,nan_idxs] = np.interp(x_vals, xp_vals,
                                                 yp_vals, *args)
            else:
                new_navdata[y_row,nan_idxs] = np.interp(x_vals, xp_vals,
                                                        yp_vals, *args)
        if inplace:
            return None
        return new_navdata

    def in_rows(self, rows):
        """Checks whether the given rows are in NavData.

        If the rows are not in NavData, it creates a KeyError and lists
        all non-existent rows.

        Parameters
        ----------
        rows : string or list/np.ndarray/tuple of strings
            Indexes to check whether they are rows in NavData
        """

        if isinstance(rows,str):
            rows = [rows]
        if isinstance(rows, (list, np.ndarray, tuple)):
            if isinstance(rows,np.ndarray):
                rows = np.atleast_1d(rows)
            missing_rows = ["'"+row+"'" for row in rows
                            if row not in self.rows]
        else:
            raise KeyError("input to in_rows must be a single row " \
                         + "index or list/np.ndarray/tuple of indexes")

        if len(missing_rows) > 0:
            raise KeyError(", ".join(missing_rows) + " row(s) are" \
                           + " missing from NavData object.")

    def find_wildcard_indexes(self, wildcards, max_allow = None,
                              excludes = None):
        """Searches for indexes matching wildcard search input.

        For example, a search for ``x_*_m`` would find ``x_rx_m`` or
        ``x_sv_m`` or ``x_alpha_beta_gamma_m`` depending on the rows
        existing in the NavData instance.

        The ``excludes`` variable allows you to exclude indexes when
        trying to match a wildcard. For example, if there are rows named
        ``pr_raw_m``and ``pr_raw_sigma_m`` then the input
        ``wildcards="pr_*_m", excludes=None`` would return
        ``{"pr_*_m", ["pr_raw_m","pr_raw_sigma_m"]}`` but with the excludes
        parameter set, the input ``wildcards="pr_*_m", excludes="pr_*_sigma_m"``
        would only return ``{"pr_*_m", ["pr_raw_m"]}``

        Will return an error no index is found matching the wildcard or
        if more than ``max_allow`` indexes are found.

        Currently only allows for a single wildcard '*' per index.

        Parameters
        ----------
        wildcards : array-like or str
            List/tuple/np.ndarray/set of indexes for which to search.
        max_allow : int or None
            Maximum number of valid indexes to allow before throwing an
            error. If None, then no limit is placed.
        excludes : array-like or str
            List or string to exclude for each wildcard in wildcards.
            Must be the same length as wildcards. Allowed to include a
            wildcard '*' character but not necessary.

        Returns
        -------
        wildcard_indexes : dict
            Dictionary of the form {"search_term", [indexes,...]},

        """

        if isinstance(wildcards,str):
            wildcards = [wildcards]
        if not isinstance(wildcards, (list,tuple,np.ndarray,set)):
            raise TypeError("wildcards input in find_wildcard_indexes" \
                         +  " must be array-like or single string")
        if not (isinstance(max_allow,int) or max_allow is None):
            raise TypeError("max_allow input in find_wildcard_indexes" \
                          + " must be an integer or None.")
        # handle exclude types
        if isinstance(excludes,str):
            excludes = [excludes]
        if excludes is None:
            excludes = [None] * len(wildcards)
        if not isinstance(excludes, (list,tuple,np.ndarray,set)):
            raise TypeError("excludes input in find_wildcard_indexes" \
                         +  " must be array-like, single string, " \
                         + "or None for each wildcard")
        if len(excludes) != len(wildcards):
            raise TypeError("excludes input must match length of " \
                          + "wildcard input.")
        for ex_idx, exclude in enumerate(excludes):
            if exclude is None or isinstance(exclude,str):
                excludes[ex_idx] = [exclude]
            if not isinstance(excludes[ex_idx], (list,tuple,np.ndarray,set)):
                raise TypeError("excludes input in find_wildcard_indexes" \
                             +  " must be array-like, single string, " \
                             + "or None for each wildcard")

        wildcard_indexes = {}

        for wild_idx, wildcard in enumerate(wildcards):
            if not isinstance(wildcard,str):
                raise TypeError("wildcards must be strings")
            if wildcard.count("*") != 1:
                raise RuntimeError("One wildcard '*' and only one "\
                          + "wildcard must be present in search string")
            indexes = [row for row in self.rows
                   if row.startswith(wildcard.split("*",maxsplit=1)[0])
                    and row.endswith(wildcard.split("*",maxsplit=1)[1])]
            if excludes[wild_idx] is not None:
                for exclude in excludes[wild_idx]:
                    if exclude is not None:
                        if '*' in exclude:
                            indexes = [row for row in indexes
                                     if not (row.startswith(exclude.split("*",maxsplit=1)[0])
                                     and row.endswith(exclude.split("*",maxsplit=1)[1]))]
                        else:
                            indexes = [row for row in indexes if exclude != row]
            if max_allow is not None and len(indexes) > max_allow:
                raise KeyError("More than " + str(max_allow) \
                             + " possible row indexes for "  + wildcard)
            if len(indexes) == 0:
                raise KeyError("Missing " + wildcard + " row.")

            wildcard_indexes[wildcard] = indexes

        return wildcard_indexes

    def pandas_df(self):
        """Return pandas DataFrame equivalent to class

        Returns
        -------
        df : pd.DataFrame
            DataFrame with data, including strings as strings
        """

        df_list = []
        for row in self.rows:
            df_list.append(np.atleast_1d(
                           self[row].astype(self.orig_dtypes[row])))

        # transpose list to conform to Pandas input
        df_list = [list(x) for x in zip(*df_list)]

        dframe = pd.DataFrame(df_list,columns=self.rows)

        return dframe

    def to_csv(self, output_path="navdata.csv", index=False, **kwargs): #pragma: no cover
        """Save data as csv

        Parameters
        ----------
        output_path : string
            Path where csv should be saved
        index : bool
            If True, will write csv row names (index).
        header : bool
            If True (default), will list out names as columns
        sep : string
            Delimiter string of length 1, defaults to ‘,’

        """
        pd_df = self.pandas_df()
        pd_df.to_csv(output_path, index=index, **kwargs)

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
    def _row_idx_str_bool(self):
        """Dictionary of index : if data entry is string.

        Row has string values if the string map is nonempty for a
        given row.

        Returns
        -------
        _row_idx_str_bool : Dict
            Dictionary of whether data at row number key is string or not
        """
        _row_idx_str_bool = {self.map[k]: bool(len(self.str_map[k])) for k in self.str_map}
        return _row_idx_str_bool

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
            columns. The return is squeezed meaning that all dimensions
            of the output that are length of one are removed
        """
        rows, cols = self._parse_key_idx(key_idx)
        row_list, row_str = self._get_str_rows(rows)
        assert np.all(row_str) or np.all(np.logical_not(row_str)), \
                "Cannot assign/return combination of strings and numbers"
        if np.all(row_str):
            # Return sliced strings
            arr_slice = np.atleast_2d(np.empty_like(self.array[rows, cols], dtype=object))
            for row_num, row in enumerate(row_list):
                str_arr = self._get_strings(self.inv_map[row])
                arr_slice[row_num, :] = str_arr[cols]
        else:
            arr_slice = self.array[rows, cols]

        if isinstance(rows,list) and len(rows) == 1 \
        and self.inv_map[rows[0]] in self.orig_dtypes:
            arr_slice = arr_slice.astype(self.orig_dtypes[self.inv_map[rows[0]]])

        # remove all dimensions of length one
        arr_slice = np.squeeze(arr_slice)

        return arr_slice

    def __setitem__(self, key_idx, new_value):
        """Add/update rows.

        __setitem__ expects that the shape of the value being passed
        matches the shape of the internal arrays that have to be set.
        So, if 2 variable types (rows) at 10 instances (columns) need to
        be set, the input new_value must be of shape (2, 10).
        If the shape is (10, 2) the assignment operator will raise a
        ValueError. This applies to all types of value assignments.

        Parameters
        ----------
        key_idx : str/list/tuple/slice/int
            Query for array items to set
        new_value : np.ndarray/list/int
            Values to be added to self.array attribute

        """
        if isinstance(key_idx, int) and len(self.map)<=key_idx:
            raise KeyError('Row indices must be strings when assigning new values')
        if isinstance(key_idx, slice) and len(self.map)==0:
            raise KeyError('Row indices must be strings when assigning new values')
        if isinstance(key_idx, str) and key_idx not in self.map:
            #Creating an entire new row
            if isinstance(new_value, np.ndarray) \
            and (new_value.dtype in (object,str) \
            or np.issubdtype(new_value.dtype,np.dtype('U'))):
                # Adding string values
                new_value = new_value.astype(str)
                new_str_vals = len(np.unique(new_value))*np.ones(np.shape(new_value),
                                    dtype=self.arr_dtype)
                new_str_vals = self._str_2_val(new_str_vals, new_value, key_idx)
                if self.array.shape == (0,0):
                    # if empty array, start from scratch
                    self.array = np.reshape(new_str_vals, [1, -1])
                else:
                    # if array is not empty, add to it
                    self.array = np.vstack((self.array, np.reshape(new_str_vals, [1, -1])))
                self.map[key_idx] = self.shape[0]-1
                # update original dtype in case of replacing values
                self.orig_dtypes[key_idx] = object
            else:
                # numeric values
                if not type(new_value) in (int,float) \
                and ((not isinstance(new_value, list) and new_value.size > 0)
                or (isinstance(new_value, list) and len(new_value) > 0)):
                    assert not isinstance(np.asarray(new_value).item(0), str), \
                            "Cannot set a row with list of strings, \
                            please use np.ndarray with dtype=object"
                # Adding numeric values
                self.str_map[key_idx] = {}
                # update original dtype in case of replacing values
                dtype = np.asarray(new_value).dtype
                if np.issubdtype(dtype, np.integer):
                    dtype = np.int64
                self.orig_dtypes[key_idx] = dtype
                if self.array.shape == (0,0):
                    # if empty array, start from scratch
                    self.array = np.reshape(new_value, (1,-1))
                    # have to explicitly convert to float in case
                    # numbers were interpretted as integers
                    self.array = self.array.astype(self.arr_dtype)
                else:
                    # if array is not empty, add to it
                    self.array = np.vstack((self.array, np.empty([1, len(self)])))
                    self.array[-1, :] = np.reshape(new_value, -1)
                self.map[key_idx] = self.shape[0]-1
        else:
            # Updating existing rows or columns
            rows, cols = self._parse_key_idx(key_idx)
            row_list, row_str = self._get_set_str_rows(rows,new_value)
            assert np.all(row_str) or np.all(np.logical_not(row_str)), \
                "Cannot assign/return combination of strings and numbers"
            if np.all(row_str):
                assert isinstance(new_value, np.ndarray) \
                and (new_value.dtype in (object,str) \
                or np.issubdtype(new_value.dtype,np.dtype('U'))), \
                    "String assignment only supported for ndarray of type object"
                inv_map = self.inv_map
                new_value = np.atleast_2d(new_value)
                new_str_vals = np.ones_like(new_value, dtype=self.arr_dtype)
                for row_num, row in enumerate(row_list):
                    key = inv_map[row]
                    new_value_row = new_value[row_num , :]
                    new_str_vals_row = new_str_vals[row_num, :]
                    new_str_vals[row_num, :] = self._str_2_val(new_str_vals_row,
                                                    new_value_row, key)
                self.array[rows, cols] = new_str_vals
                # update original dtype in case of replacing values
                for row_index in rows:
                    self.orig_dtypes[self.inv_map[row_index]] = object
            else:
                if not isinstance(new_value, int):
                    assert not isinstance(np.asarray(new_value).item(0), str), \
                            "Please use dtype=object for string assignments"
                self.array[rows, cols] = new_value
                # update original dtype in case of replacing values
                for row_index in rows:
                    dtype = np.asarray(new_value).dtype
                    if np.issubdtype(dtype, np.integer):
                        dtype = np.int64
                    self.orig_dtypes[self.inv_map[row_index]] = dtype

    def __iter__(self):
        """Initialize iterator over NavData (iterates over all columns)

        Returns
        -------
        self: gnss_lib_py.parsers.NavData
            Instantiation of NavData class with iteration initialized
        """
        self.curr_col = 0
        self.num_cols = np.shape(self.array)[1]
        return self

    def __next__(self):
        """Method to get next item when iterating over NavData class

        Returns
        -------
        x_curr : gnss_lib_py.parsers.NavData
            Current column (based on iteration count)
        """
        if self.curr_col >= self.num_cols:
            raise StopIteration
        x_curr = self.copy(rows=None, cols=self.curr_col)
        self.curr_col += 1
        return x_curr

    def __str__(self):
        """Creates string representation of NavData object

        Returns
        -------
        str_out : str
            String representation of Navdata object, based on equivalent
            Pandas string
        """
        str_out = str(self.pandas_df())
        str_out = str_out.replace("DataFrame","NavData")
        str_out = str_out.replace("Columns","Rows")
        # swap rows and columns when printing for NavData consistency
        str_out = re.sub(r"(.*)\[(\d+)\srows\sx\s(\d+)\scolumns\](.*)",
                         r'\g<1>[\g<3> rows x \g<2> columns]\g<4>',
                         str_out)
        return str_out

    def __repr__(self):  # pragma: no cover
        """Evaluated string representation of Navdata object

        For NavData objects, this is similar to the str method and is
        defined separately to avoid having to add a `print` method
        before each display command in Jupyter notebooks

        Returns
        -------
        rep_out : str
            Evaluated string representation object
        """
        rep_out = str(self)
        return rep_out

    def __len__(self):
        """Return length of class

        Returns
        -------
        length : int
            Number of time steps in NavData
        """
        length = np.shape(self.array)[1]
        return length

    def _build_navdata(self):
        """Build attributes for NavData.

        """
        self.array = np.zeros((0,0), dtype=self.arr_dtype)

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
        _row_idx_str_bool = self._row_idx_str_bool
        if isinstance(rows, slice):
            slice_idx = rows.indices(self.shape[0])
            row_list = np.arange(slice_idx[0], slice_idx[1], slice_idx[2])
            row_str = [_row_idx_str_bool[row] for row in row_list]
        else:
            row_list = list(rows)
            row_str = [_row_idx_str_bool[row] for row in rows]
        return row_list, row_str

    def _get_set_str_rows(self, rows, new_value):
        """Checks which output rows contain string elements given input.

        If the row used to be string values but now is numeric, this
        function also empties the corresponding dictionary in str_map.

        Parameters
        ----------
        rows : slice/list
            Rows to check for string elements
        new_value : np.ndarray/list/int
            Values to be added to self.array attribute

        Returns
        -------
        row_list : list
            Input rows, strictly in list format
        row_str_new : list
            List of boolean values indicating which of the new rows
            contain strings.
        """

        row_list, row_str_existing = self._get_str_rows(rows)

        if isinstance(new_value, np.ndarray) and (new_value.dtype in (object,str) \
                        or np.issubdtype(new_value.dtype,np.dtype('U'))):
            if isinstance(new_value.item(0), (int, float)):
                row_str_new = [False]*len(row_list)
            else:
                row_str_new = [True]*len(row_list)
        elif isinstance(np.asarray(new_value).item(0), str):
            raise RuntimeError("Cannot set a row with list of strings, \
                             please use np.ndarray with dtype=object")
        else:
            row_str_new = [False]*len(row_list)

        for row_idx, row in enumerate(row_list):
            if row_str_existing[row_idx] and not row_str_new[row_idx]:
                # changed from string to numeric
                self.str_map[self.inv_map[row]] = {}

        return row_list, row_str_new

    def _str_2_val(self, new_str_vals, new_value, key):
        """Convert string valued arrays to values for storing in array

        Parameters
        ----------
        new_str_vals : np.ndarray
            Array of dtype=self.arr_dtype where numeric values are to be
            stored
        new_value : np.ndarray
            Array of dtype=object, containing string values that are to
            be converted
        key : string
            Key indicating row where string to numeric conversion is
            required
        """
        if key in self.map:
            # Key already exists, update existing string value dictionary
            inv_str_map = {v: k for k, v in self.str_map[key].items()}
            string_vals = np.unique(new_value)
            str_map_dict = self.str_map[key]
            total_str = len(self.str_map[key])
            for str_val in string_vals:
                if str_val not in inv_str_map.keys():
                    str_map_dict[total_str] = str_val
                    new_str_vals[new_value==str_val] = total_str
                    total_str += 1
                else:
                    new_str_vals[new_value==str_val] = inv_str_map[str_val]
            self.str_map[key] = str_map_dict
        else:
            string_vals = np.unique(new_value)
            str_dict = dict(enumerate(string_vals))
            self.str_map[key] = str_dict
            new_str_vals = len(string_vals)*np.ones(np.shape(new_value),
                                                   dtype=self.arr_dtype)
            # Set unassigned value to int not accessed by string map
            for str_key, str_val in str_dict.items():
                if new_str_vals.size == 1:
                    new_str_vals = np.array(str_key,dtype=self.arr_dtype)
                else:
                    new_str_vals[new_value==str_val] = str_key
            # Copy set to false to prevent memory overflows
            new_str_vals = np.round(new_str_vals.astype(self.arr_dtype,
                                                        copy=False))
        return new_str_vals

    def _get_strings(self, key):
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
            self.in_rows(key_idx)
            rows = [self.map[key_idx]]
            cols = slice(None, None)
        elif isinstance(key_idx, list) and isinstance(key_idx[0], str):
            rows = [self.map[k] for k in key_idx]
            cols = slice(None, None)
        elif isinstance(key_idx, slice):
            rows = key_idx
            cols = slice(None, None)
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
        return rows, cols

    @staticmethod
    def _row_map():
        """Map of column names from loaded to gnss_lib_py standard

        Initializes as an emptry dictionary, must be reimplemented for
        custom parsers.

        Returns
        -------
        row_map : Dict
            Dictionary of the form {old_name : new_name}
        """

        row_map = {}

        return row_map
