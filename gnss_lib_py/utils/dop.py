"""
Dillution of Precision (DOP) calculations and interface with NavData.
"""

__authors__ = "Ashwin Kanhere, Daniel Neamati"
__date__ = "24 January, 2024"


import numpy as np

from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.navdata.operations import loop_time
from gnss_lib_py.utils.coordinates import el_az_to_enu_unit_vector


def get_enu_dop_labels():
    """
    Helper function to get the DOP labels.

    Returns
    -------
    dop_labels : list
        List of strings for the DOP labels.
    """

    dop_labels = ['ee', 'en', 'eu', 'et',
                        'nn', 'nu', 'nt',
                              'uu', 'ut',
                                    'tt']
    return dop_labels


def get_dop(navdata, **which_dop):
    """Get DOP from navdata.

    Parameters
    ----------
    navdata : gnss_lib_py.navdata.navdata.NavData
        NavData instance containing received GNSS measurements for a
        particular time instance, contains elevation and azimuth angle
        information for an estimated location.
        Instance of the NavData class which must include at least
        ``gps_millis``, ``el_sv_deg``, and ``az_sv_deg``

    which_dop : dict
        Dictionary of which dop values are needed. Default is HDOP and VDOP.

        Note that the dop matrix output is splatted across entries following
        the behavior below:
        ::

            [[EE, EN, EU, ET],
             [NE, NN, NU, NT],
             [UE, UN, UU, UT],
             [TE, TN, TU, TT]]  (16 values in 2D array)

        is stored as:
        ::

             [EE, EN, EU, ET,
                  NN, NU, NT,
                      UU, UT,
                          TT] (10 values in 1D array)

        recognizing that the dop matrix is symmetric.

    Returns
    -------
    dop_navdata : gnss_lib_py.navdata.navdata.NavData
        Dilution of precision data along with the relevant time stamp.
    """

    # Check that the navdata has the necessary elevation and azimuth data
    navdata.in_rows(['gps_millis', 'el_sv_deg', 'az_sv_deg'])

    # Default which_dop values assume HDOP and VDOP are needed.
    default_which_dop = {'GDOP': False,
                         'HDOP': True,
                         'VDOP': True,
                         'PDOP': False,
                         'TDOP': False,
                         'dop_matrix': False}
    # This syntax allows for the user to override the default values.
    which_dop = {**default_which_dop, **which_dop}

    # Initialize the gps_millis to output
    times = []

    # Initialize the dop to output
    dop_out = {}
    for dop_name, include_dop in which_dop.items():
        if include_dop:
            dop_out[dop_name] = []

    # Loop through time in the navdata.
    for timestamp, _, navdata_subset in loop_time(navdata, 'gps_millis'):
        # Append the timestamp
        times.append(timestamp)

        # remove NaN indices
        nan_indices = np.where(np.isnan(navdata_subset['el_sv_deg']) \
                               | np.isnan(navdata_subset['az_sv_deg']))[0]
        navdata_subset = navdata_subset.remove(None, nan_indices)

        # Calculate the DOP at this time instance
        dop = calculate_dop(navdata_subset)

        # Append the DOP to the output
        for dop_name, include_dop in which_dop.items():
            if include_dop:
                dop_out[dop_name].append(dop[dop_name])

    # Create a new NavData instance to store the DOP
    dop_navdata = NavData()
    dop_navdata['gps_millis'] = np.array(times)

    for dop_name, include_dop in which_dop.items():
        # We need to handle the dop_matrix separately
        if include_dop and dop_name != 'dop_matrix':
            dop_navdata[dop_name] = np.array(dop_out[dop_name])

    # Special handling for splatting the dop_matrix
    if which_dop['dop_matrix']:

        dop_labels = get_enu_dop_labels()

        dop_matrix_splat = []

        for dop_matrix in dop_out['dop_matrix']:
            dop_matrix_splat.append(splat_dop_matrix(dop_matrix))

        # Convert entire array across time to numpy array
        dop_matrix_splat = np.array(dop_matrix_splat)
        assert dop_matrix_splat.shape == (len(times), len(dop_labels)), \
            f"DOP matrix splatted to {dop_matrix_splat.shape}."

        # Add to the NavData instance
        for dop_ind, dop_label in enumerate(dop_labels):
            dop_navdata[f'dop_{dop_label}'] = dop_matrix_splat[:, dop_ind]

    return dop_navdata


def splat_dop_matrix(dop_matrix):
    """
    Splat the DOP matrix into a 1D array. Note that the dop matrix output
    is splatted across entries following
    the behavior below:
    ::

        [[EE, EN, EU, ET],
         [NE, NN, NU, NT],
         [UE, UN, UU, UT],
         [TE, TN, TU, TT]]  (16 values in 2D array)

    is stored as:
    ::

         [EE, EN, EU, ET,
              NN, NU, NT,
                  UU, UT,
                      TT] (10 values in 1D array)

    recognizing that the dop matrix is symmetric.

    Parameters
    ----------
    dop_matrix : np.ndarray
        DOP matrix in ENU coordinates of size (4, 4).

    Returns
    -------
    dop_splat : np.ndarray
        DOP matrix splatted into a 1D array of size (10,).
    """

    # Splat the DOP matrix
    dop_splat = dop_matrix[(0, 0, 0, 0, 1, 1, 1, 2, 2, 3),
                           (0, 1, 2, 3, 1, 2, 3, 2, 3, 3)]

    return np.array(dop_splat)


def unsplat_dop_matrix(dop_splat):
    """
    Un-splat the DOP matrix from a 1D array. Note that the dop matrix output
    is unsplatted across entries following the behavior below:
    ::

        [EE, EN, EU, ET,
             NN, NU, NT,
                 UU, UT,
                     TT] (10 values in 1D array)

    is unsplatted to
    ::

        [[EE, EN, EU, ET],
         [NE, NN, NU, NT],
         [UE, UN, UU, UT],
         [TE, TN, TU, TT]]  (16 values in 2D array)

    recognizing that the dop matrix is symmetric.

    Parameters
    ----------
    dop_splat : np.ndarray
        DOP matrix splatted into a 1D array of size (10,).

    Returns
    -------
    dop_matrix : np.ndarray
        DOP matrix in ENU coordinates of size (4, 4).
    """

    # Un-splat the DOP matrix
    dop_matrix = np.zeros((4, 4))

    splat_rows = (0, 0, 0, 0, 1, 1, 1, 2, 2, 3)
    splat_cols = (0, 1, 2, 3, 1, 2, 3, 2, 3, 3)

    # Fill in the upper triangle of the DOP matrix
    dop_matrix[splat_rows, splat_cols] = dop_splat
    # Fill in the lower triangle of the DOP matrix
    # (Note that the diagonal is filled in again, but that's okay.)
    dop_matrix[splat_cols, splat_rows] = dop_splat

    return dop_matrix


def calculate_enu_dop_matrix(derived):
    """
    Calculate the DOP matrix from elevation and azimuth (ENU).

    Parameters
    ----------
    derived : gnss_lib_py.navdata.navdata.NavData
        NavData instance containing received GNSS measurements for a
        particular time instance, contains elevation and azimuth angle
        information for an estimated location.

    Returns
    -------
    dop_matrix : np.ndarray
        DOP matrix of size (4, 4).
    """

    # Use the elevation and azimuth angles to get the ENU and Time matrix
    # Each row is [d_e, d_n, d_u, 1] for each satellite.
    enut_matrix = _calculate_enut_matrix(derived)
    enut_gram_matrix = enut_matrix.T @ enut_matrix

    # Calculate the DOP matrix
    try:
        dop_matrix = np.linalg.inv(enut_gram_matrix)
    except np.linalg.LinAlgError:
        # If the matrix is singular, return NaNs for the DOP
        dop_matrix = np.nan * np.ones((4, 4))

    return dop_matrix


def parse_dop(dop_matrix):
    """Calculate DOP types from the DOP matrix.

    Parameters
    ----------
    dop_matrix : np.ndarray
        DOP matrix in ENU coordinates of size (4, 4).

    Returns
    -------
    dop : Dict
        Dilution of precision, with DOP type as the keys: "HDOP", "VDOP",
        "TDOP", "PDOP", "GDOP".
    """

    dop = {}
    dop["dop_matrix"] = dop_matrix

    dop["GDOP"] = _safe_sqrt(np.trace(dop_matrix))
    dop["HDOP"] = _safe_sqrt(dop_matrix[0, 0] + dop_matrix[1, 1])
    dop["VDOP"] = _safe_sqrt(dop_matrix[2, 2])
    dop["PDOP"] = _safe_sqrt(dop_matrix[0, 0] + \
                             dop_matrix[1, 1] + \
                             dop_matrix[2, 2])
    dop["TDOP"] = _safe_sqrt(dop_matrix[3, 3])

    return dop


def _safe_sqrt(x):
    """
    Safe square root for DOP calculations.

    Parameters
    ----------
    x : float
        Value to take the square root of.

    Returns
    -------
    y : float
        Square root of x, or NaN if x is negative.
    """
    return np.sqrt(x) if x >= 0 else np.nan


def calculate_dop(derived):
    """
    Calculate the DOP from elevation and azimuth (ENU).

    Parameters
    ----------
    derived : gnss_lib_py.navdata.navdata.NavData
        NavData instance containing received GNSS measurements for a
        particular time instance, contains elevation and azimuth angle
        information for an estimated location.

    Returns
    -------
    dop : Dict
        Dilution of precision, with DOP type as the keys: "HDOP", "VDOP",
        "TDOP", "PDOP", "GDOP".
    """

    # Calculate the DOP matrix
    dop_matrix = calculate_enu_dop_matrix(derived)

    # Parse the DOP matrix to get the DOP values
    dop = parse_dop(dop_matrix)

    return dop


def _calculate_enut_matrix(derived):
    """
    Calculate the ENU and Time Matrix from elevation and azimuth.
    Each row is [d_e, d_n, d_u, 1] for each satellite.

    Parameters
    ----------
    derived : gnss_lib_py.navdata.navdata.NavData
        NavData instance containing received GNSS measurements for a
        particular time instance, contains elevation and azimuth angle
        information for an estimated location.

    Returns
    -------
    enut_matrix : np.ndarray
        Matrix of ENU and Time vectors of size (num_satellites, 4).

    """
    enu_unit_dir_mat = el_az_to_enu_unit_vector(derived['el_sv_deg'],
                                                derived['az_sv_deg'])
    enut_matrix = np.hstack((enu_unit_dir_mat,
                             np.ones((enu_unit_dir_mat.shape[0], 1))))

    return enut_matrix
