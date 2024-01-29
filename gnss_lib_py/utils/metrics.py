"""Metrics to quantify quality of state estimates or GNSS measurements.
"""

__authors__ = "Ashwin Kanhere, Daniel Neamati"
__date__ = "24 January, 2024"


import numpy as np
from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.navdata.operations import find_wildcard_indexes



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
    unit_dir_mat = np.vstack((np.atleast_2d(np.cos(sv_el_az_rad[0,:]) * np.sin(sv_el_az_rad[1,:])),
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


def accuracy_statistics(state_estimate, ground_truth, est_type="pos", statistic="mean"):
    """Calculate required statistic for accuracy of state estimate.

    Parameters
    ----------
    state_estimate : gnss_lib_py.navdata.navdata.NavData
        NavData instance containing state estimates.
    ground_truth : gnss_lib_py.navdata.navdata.NavData
        NavData instance containing ground truth corresponding to state
        estimate. Must have same time stamps as state_estimate.
    est_type : str, optional
        Type of state estimate, by default "pos". Can be "pos", "vel",
        "time" or "acc".
    statistic : str, optional
        Statistic to calculate, by default "mean". Can be "mean",
        "median", "std", "max" or quantiles.
    """
    if est_type == "pos":
        row_wildcards = ["x_rx*_m", "y_rx*_m", "z_rx*_m"]
    elif est_type == "vel":
        row_wildcards = ["vx_rx*_mps", "vy_rx*_mps", "vz_rx*_mps"]
    elif est_type == "time":
        row_wildcards = ["b_rx*_m", "b_dot_rx*_mps"]
    elif est_type == "acc":
        row_wildcards = ["ax_rx*_mps2", "ay_rx*_mps2", "az_rx*_mps2"]
    else:
        raise ValueError("Invalid est_type: {}".format(est_type))

    # Extract subsets of the state_estimate and ground_truth NavData
    state_estimate_rows = find_wildcard_indexes(state_estimate, row_wildcards)
    state_estimate_subset = state_estimate

    error_values = _get_err_values(state_estimate_subset, ground_truth_subset)
