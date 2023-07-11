"""Fault Detection and Exclusion (FDE) methods for GNSS applications.

"""

__authors__ = "D. Knowles"
__date__ = "11 Jul 2023"


def solve_fde(navdata, method="residual", remove_outliers=False):
    """Detects and optionally removes GNSS measurement faults.

    Individual fault detection and exclusion (fde) methods are
    documented in their individual functions.

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
        NavData of GNSS measurements which must include the receiver's
        estimated state: x_rx*_m, y_rx*_m, z_rx*_m, b_rx*_m as well as
        the satellite states: x_sv_m, y_sv_m, z_sv_m, b_sv_m as well
        as timing "gps_millis" and the corrected pseudorange corr_pr_m.
    method : string
        Method for fault detection and exclusion either "residual" for
        residual-based, "ss" for solution separation or "edm" for
        Euclidean Distance Matrix-based.

    Returns
    -------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Result includes a new row of ``fault_<method>`` which has a
        boolean 0/1 value where 1 indicates a fault and 0 indicates that
        no fault was detected.

    """

    if method == "residual":
        navdata = fde_residual(navdata)
    elif method == "ss":
        navdata = fde_solution_separation(navdata)
    elif method == "edm":
        navdata = fde_edm(navdata)
    else:
        raise ValueError("invalid method input for solve_fde()")

    if remove_outliers:
        navdata = navdata.where("fault_"+method,False)

    return navdata


def fde_residual(navdata):
    """Residual-based fault detection and exclusion.

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
        NavData of GNSS measurements which must include the receiver's
        estimated state: x_rx*_m, y_rx*_m, z_rx*_m, b_rx*_m as well as
        the satellite states: x_sv_m, y_sv_m, z_sv_m, b_sv_m.

    Returns
    -------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Result includes a new row of ``fault_residual`` which has a
        boolean 0/1 value where 1 indicates a fault and 0 indicates that
        no fault was detected.

    """

    navdata["fault_residual"] = 0

    return navdata

def fde_solution_separation(navdata):
    """Solution separation fault detection and exclusion.

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
        NavData of GNSS measurements which must include the receiver's
        estimated state: x_rx*_m, y_rx*_m, z_rx*_m, b_rx*_m as well as
        the satellite states: x_sv_m, y_sv_m, z_sv_m, b_sv_m.

    Returns
    -------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Result includes a new row of ``fault_ss`` which has a
        boolean 0/1 value where 1 indicates a fault and 0 indicates that
        no fault was detected.

    """

    navdata["fault_ss"] = 0

    return navdata

def fde_edm(navdata):
    """Euclidean distance matrix-based fault detection and exclusion.

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
        NavData of GNSS measurements which must include the receiver's
        estimated state: x_rx*_m, y_rx*_m, z_rx*_m, b_rx*_m as well as
        the satellite states: x_sv_m, y_sv_m, z_sv_m, b_sv_m.

    Returns
    -------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Result includes a new row of ``fault_edm`` which has a
        boolean 0/1 value where 1 indicates a fault and 0 indicates that
        no fault was detected.

    """

    navdata["fault_edm"] = 0

    return navdata
