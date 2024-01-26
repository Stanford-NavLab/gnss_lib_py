"""Metrics to quantify quality of state estimates or GNSS measurements.
"""

__authors__ = "Ashwin Kanhere, Daniel Neamati"
__date__ = "24 January, 2024"


import numpy as np


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