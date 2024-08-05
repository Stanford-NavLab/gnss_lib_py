
"""Base class for moving values between different modules/functions

"""

__authors__ = "Ashwin Kanhere, D. Knowles"
__date__ = "03 Nov 2021"

import numpy as np

from gnss_lib_py.navdata.navdata import NavData

def concat(*navdatas, axis=1):
    """Concatenates NavData instances by row or column.

    Concatenates given NavData instances together by either row or
    column.

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
    navdatas : List-like of gnss_lib_py.navdata.navdata.NavData
        Navdata instances to concatenate.
    axis : int
        Either add new rows (type) of data ``axis=0`` or new columns
        (e.g. timesteps) of data ``axis=1``.

    Returns
    -------
    new_navdata : gnss_lib_py.navdata.navdata.NavData or None
        NavData instance after concatenating specified data.

    """

    concat_navdata = navdatas[0].copy()
    for navdata in navdatas[1:]:
        if not isinstance(navdata,NavData):
            raise TypeError("concat input data must be a NavData instance.")

        if axis == 0: # concatenate new rows
            if len(concat_navdata) != len(navdata):
                raise RuntimeError("concat input data must be same " \
                                 + "length to concatenate new rows.")

            for row in navdata.rows:
                new_row_name = row
                suffix = None
                while new_row_name in concat_navdata.rows:
                    if suffix is None:
                        suffix = 0
                    else:
                        suffix += 1
                    new_row_name = row + "_" + str(suffix)
                new_row = navdata[row].astype(navdata.orig_dtypes[row])
                concat_navdata[new_row_name] = new_row

        elif axis == 1: # concatenate new columns
            new_navdata = NavData()
            # get unique list of row names
            combined_rows = concat_navdata.rows  + [row for row in navdata.rows
                                          if row not in concat_navdata.rows]

            for row in combined_rows:
                combined_row = np.array([])
                # combine data from existing and new instance
                for data in [concat_navdata, navdata]:
                    if row in data.rows:
                        new_row = np.atleast_1d(data[row])
                    elif len(data) == 0:
                        continue
                    else:
                        # add np.nan for missing values
                        new_row = np.empty((len(data),))
                        new_row.fill(np.nan)
                    new_row = np.array(new_row, ndmin=1)
                    combined_row = np.concatenate((combined_row,
                                                   new_row))
                new_navdata[row] = combined_row

            if len(concat_navdata) > 0 and len(navdata) > 0:
                new_navdata.orig_dtypes = navdata.orig_dtypes.copy()
                new_navdata.orig_dtypes.update(concat_navdata.orig_dtypes)
            elif len(concat_navdata) > 0:
                new_navdata.orig_dtypes = concat_navdata.orig_dtypes.copy()
            elif len(navdata) > 0:
                new_navdata.orig_dtypes = navdata.orig_dtypes.copy()

            concat_navdata.array = new_navdata.array
            concat_navdata.map = new_navdata.map
            concat_navdata.str_map = new_navdata.str_map
            concat_navdata.orig_dtypes = new_navdata.orig_dtypes.copy()

    return concat_navdata

def sort(navdata, order=None, ind=None, ascending=True,
         inplace=False):
    """Sort values along given row or using given index

    Parameters
    ----------
    navdata : gnss_lib_py.navdata.navdata.NavData
        Navdata instance.
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
    new_navdata : gnss_lib_py.navdata.navdata.NavData or None
        If inplace is False, returns NavData instance after renaming
        specified rows. If inplace is True, returns
        None.

    """
    # check if there is only one column - no sorting needed
    if navdata.shape[1] == 1:
        return navdata

    if ind is None:
        assert order is not None, \
        "Provide 'order' arg as row on which NavData is sorted"
        if ascending:
            ind = np.argsort(navdata[order])
        else:
            ind = np.argsort(-navdata[order])

    if not inplace:
        new_navdata = navdata.copy()   # create copy to return
    for row_idx in range(navdata.shape[0]):
        if inplace:
            navdata.array[row_idx,:] = navdata.array[row_idx,ind]
        else:
            new_navdata.array[row_idx,:] = new_navdata.array[row_idx,ind] # pylint: disable=possibly-used-before-assignment

    if inplace:
        return None
    return new_navdata

def loop_time(navdata, time_row, delta_t_decimals=2):
    """Generator object to loop over columns from same times.

    Parameters
    ----------
    navdata : gnss_lib_py.navdata.navdata.NavData
        Navdata instance.
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
    new_navdata : gnss_lib_py.navdata.navdata.NavData
        NavData with same time, up to given decimal tolerance.

    """

    times = navdata[time_row]
    times_unique = np.sort(np.unique(np.around(times,
                                     decimals=delta_t_decimals)))
    for time_idx, time in enumerate(times_unique):
        if time_idx==0:
            delta_t = 0
        else:
            delta_t = time-times_unique[time_idx-1]
        new_navdata = navdata.where(time_row, [time-10**(-delta_t_decimals),
                                            time+10**(-delta_t_decimals)],
                                            condition="between")
        if len(np.unique(new_navdata[time_row]))==1:
            frame_time = new_navdata[time_row, 0]
        else:
            frame_time = time
        yield frame_time, delta_t, new_navdata

def interpolate(navdata, x_row, y_rows, inplace=False, *args):
    """Interpolate NaN values based on row data.

    Additional ``*args`` arguments are passed into the ``np.interp``
    function.

    Parameters
    ----------
    navdata : gnss_lib_py.navdata.navdata.NavData
        Navdata instance.
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
    new_navdata : gnss_lib_py.navdata.navdata.NavData or None
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
    navdata.in_rows([x_row] + y_rows)

    if not inplace:
        new_navdata = navdata.copy()
    for y_row in y_rows:
        nan_idxs = navdata.argwhere(y_row,np.nan)
        if nan_idxs.size == 0:
            continue
        not_nan_idxs = navdata.argwhere(y_row,np.nan,"neq")
        x_vals = navdata[x_row,nan_idxs]
        xp_vals = navdata[x_row,not_nan_idxs]
        yp_vals = navdata[y_row,not_nan_idxs]

        if inplace:
            navdata[y_row,nan_idxs] = np.interp(x_vals, xp_vals,
                                             yp_vals, *args)
        else:
            new_navdata[y_row,nan_idxs] = np.interp(x_vals, xp_vals, # pylint: disable=possibly-used-before-assignment
                                                    yp_vals, *args)
    if inplace:
        return None
    return new_navdata

def find_wildcard_indexes(navdata, wildcards, max_allow = None,
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
    navdata : gnss_lib_py.navdata.navdata.NavData
        Navdata instance.
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
        indexes = [row for row in navdata.rows
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
