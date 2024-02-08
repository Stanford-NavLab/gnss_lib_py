"""Metrics to quantify quality of state estimates or GNSS measurements.
"""

__authors__ = "Ashwin Kanhere, Daniel Neamati"
__date__ = "24 January, 2024"


import numpy as np

from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.navdata.operations import loop_time


def get_dop(navdata, **which_dop):
    """Get DOP from navdata.

    Parameters
    ----------
    navdata : gnss_lib_py.navdata.navdata.NavData
        NavData instance containing received GNSS measurements for a
        particular time instance, contains elevation and azimuth angle
        information for an estimated location.

    which_dop : dict
        Dictionary of which dop values are needed. Default is HDOP and VDOP.
        
        Note that the dop matrix output is splatted across entries following 
        the behavior below:
        [[EE, EN, EU, ET],
         [NE, NN, NU, NT],
         [UE, UN, UU, UT],
         [TE, TN, TU, TT]]  (16 values in 2D array)
        is stored as
        [EE, EN, EU, ET, 
             NN, NU, NT, 
                 UU, UT, 
                     TT] (10 values in 1D array), 
        recognizing that the dop matrix is symmetric.

    Returns
    -------
    dop_navdata : gnss_lib_py.navdata.navdata.NavData
        Dilution of precision data along with the relevant time stamp.
    """

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
        dop_matrix_splat = []

        dop_labels = ['ee', 'en', 'eu', 'et', 
                            'nn', 'nu', 'nt', 
                                  'uu', 'ut', 
                                        'tt']
        
        for dop_matrix in dop_out['dop_matrix']:
            dop_matrix_splat.append(dop_matrix[(0, 0, 0, 0, 1, 1, 1, 2, 2, 3), 
                                               (0, 1, 2, 3, 1, 2, 3, 2, 3, 3)])
        
        # Convert to numpy array
        dop_matrix_splat = np.array(dop_matrix_splat)
        assert dop_matrix_splat.shape == (len(times), len(dop_labels)), \
            f"DOP matrix splatted to {dop_matrix_splat.shape}."

        # Add to the NavData instance
        for dop_ind, dop_label in enumerate(dop_labels):
            dop_navdata[f'dop_{dop_label}'] = dop_matrix_splat[:, dop_ind]

    return dop_navdata


def calculate_dop(derived):
    """Calculate DOP from state estimate.

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

    # Get the elevation and azimuth angles
    sv_el_az_rad = np.deg2rad(derived['el_sv_deg', 'az_sv_deg'])

    # Important: This matrix is in ENU coordinates, not NED.
    unit_dir_mat = np.vstack(
        (np.atleast_2d(np.cos(sv_el_az_rad[0,:]) * np.sin(sv_el_az_rad[1,:])),
         np.atleast_2d(np.cos(sv_el_az_rad[0,:]) * np.cos(sv_el_az_rad[1,:])),
         np.atleast_2d(np.sin(sv_el_az_rad[0,:])),
         np.ones((1, sv_el_az_rad.shape[1])))).T
    dop_matrix = np.linalg.inv(np.matmul(unit_dir_mat.T, unit_dir_mat))

    # Calculate the DOP
    dop = {}
    dop["dop_matrix"] = dop_matrix
    dop["GDOP"] = np.sqrt(np.trace(dop_matrix))
    dop["HDOP"] = np.sqrt(dop_matrix[0, 0] + dop_matrix[1, 1])
    dop["VDOP"] = np.sqrt(dop_matrix[2, 2])
    dop["PDOP"] = np.sqrt(dop_matrix[0, 0] + dop_matrix[1, 1] + dop_matrix[2, 2])
    dop["TDOP"] = np.sqrt(dop_matrix[3, 3])

    return dop
