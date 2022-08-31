"""Point solution methods using GNSS measurements.

This module contains point solution methods for estimating position
at a single GNSS measurement epoch. Position is solved using the
Weighted Least Squares algorithm.

"""

__authors__ = "D. Knowles, Shubh Gupta, Bradley Collicott"
__date__ = "25 Jan 2022"

import warnings

import numpy as np

from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.utils.coordinates import ecef_to_geodetic

def solve_wls(measurements, weight_type = None,
              only_bias = False, tol = 1e-7, max_count = 20):
    """Runs weighted least squares across each timestep.

    Runs weighted least squares across each timestep and adds a new
    row for the receiver's position and clock bias.

    The option for only_bias allows the user to only calculate the clock
    bias if the receiver position is already known. Only the bias term
    in rx_est_m will be updated if only_bias is set to True.

    Parameters
    ----------
    measurements : gnss_lib_py.parsers.navdata.NavData
        Instance of the NavData class
    weight_type : string
        Must either be None or the name of a row in measurements
    only_bias : bool
        If True, then only the receiver clock bias is estimated.
        Otherwise, both position and clock bias are estimated.
    tol : float
        Tolerance used for the convergence check.
    max_count : int
        Number of maximum iterations before process is aborted and
        solution returned.

    Returns
    -------
    state_estimate : gnss_lib_py.parsers.navdata.NavData
        Estimated receiver position in ECEF frame in meters and the
        estimated receiver clock bias also in meters as an instance of
        the NavData class with shape (4 x # unique timesteps) and
        the following rows: x_rx_m, y_rx_m, z_rx_m, b_rx_m.

    """

    # check that all necessary rows exist
    measurements.in_rows(["x_sv_m","y_sv_m","z_sv_m","b_sv_m",
                          "gps_millis"])

    states = []

    position = np.zeros((4,1))
    for timestamp, _, measurement_subset in measurements.loop_time("gps_millis"):

        pos_sv_m = measurement_subset[["x_sv_m","y_sv_m","z_sv_m"]].T
        pos_sv_m = np.atleast_2d(pos_sv_m)

        corr_pr_m = measurement_subset["corr_pr_m"].reshape(-1,1)

        # remove NaN indexes
        not_nan_indexes = ~np.isnan(pos_sv_m).any(axis=1)
        pos_sv_m = pos_sv_m[not_nan_indexes]
        corr_pr_m = corr_pr_m[not_nan_indexes]

        if weight_type is not None:
            if isinstance(weight_type,str) and weight_type in measurements.rows:
                weights = measurement_subset[weight_type].reshape(-1,1)
            else:
                raise TypeError("WLS weights must be None or row"\
                                +" in NavData")
        else:
            weights = None

        try:
            position = wls(position, pos_sv_m, corr_pr_m, weights,
                           only_bias, tol, max_count)

            states.append([timestamp] + np.squeeze(position).tolist())
        except RuntimeError as error:
            warnings.warn("RuntimeError encountered at gps_millis: " \
                        + str(int(timestamp)) + " RuntimeError: " \
                        + str(error), RuntimeWarning)
    states = np.array(states)

    if states.size == 0:
        warnings.warn("No valid state estimate computed in WLS, "\
                    + "returning None.", RuntimeWarning)
        return None

    state_estimate = NavData()
    state_estimate["gps_millis"] = states[:,0]
    state_estimate["x_rx_m"] = states[:,1]
    state_estimate["y_rx_m"] = states[:,2]
    state_estimate["z_rx_m"] = states[:,3]
    state_estimate["b_rx_m"] = states[:,4]

    lat,lon,alt = ecef_to_geodetic(state_estimate[["x_rx_m",
                                                   "y_rx_m",
                                                   "z_rx_m"]])
    state_estimate["lat_rx_deg"] = lat
    state_estimate["lon_rx_deg"] = lon
    state_estimate["alt_rx_deg"] = alt

    return state_estimate

def wls(rx_est_m, pos_sv_m, corr_pr_m, weights = None,
        only_bias = False, tol = 1e-7, max_count = 20):
    """Weighted least squares solver for GNSS measurements.

    The option for only_bias allows the user to only calculate the clock
    bias if the receiver position is already known. Only the bias term
    in rx_est_m will be updated if only_bias is set to True.

    Parameters
    ----------
    rx_est_m : np.ndarray
        Estimated receiver position in ECEF frame in meters and the
        estimated receiver clock bias also in meters in an
        array with shape (4 x 1) and the following order:
        x_rx_m, y_rx_m, z_rx_m, b_rx_m.
    pos_sv_m : np.ndarray
        Satellite positions as an array of shape [# svs x 3] where
        the columns contain in order x_sv_m, y_sv_m, and z_sv_m.
    corr_pr_m : np.ndarray
        Corrected pseudoranges for all satellites with shape of
        [# svs x 1]
    weights : np.array
        Weights as an array of shape [# svs x 1] where the column
        is in the same order as pos_sv_m and corr_pr_m
    only_bias : bool
        If True, then only the receiver clock bias is estimated.
        Otherwise, both position and clock bias are estimated.
    tol : float
        Tolerance used for the convergence check.
    max_count : int
        Number of maximum iterations before process is aborted and
        solution returned.

    Returns
    -------
    rx_est_m : np.ndarray
        Estimated receiver position in ECEF frame in meters and the
        estimated receiver clock bias also in meters in an
        array with shape (4 x 1) and the following order:
        x_rx_m, y_rx_m, z_rx_m, b_rx_m.

    """

    rx_est_m = rx_est_m.copy() # don't change referenced value

    count = 0
    num_svs = pos_sv_m.shape[0]
    if num_svs < 4:
        raise RuntimeError("Need at least four satellites for WLS.")
    pos_x_delta = np.inf*np.ones((4,1))

    # load weights
    if weights is None:
        weight_matrix = np.eye(num_svs)
    elif isinstance(weights, np.ndarray):
        if weights.ndim != 0 and weights.size == num_svs:
            weight_matrix = np.eye(num_svs)*weights
        else:
            raise TypeError("WLS weights must be the same length"\
                            + " as number of satellites.")
    else:
        raise TypeError("WLS weights must be None or np.ndarray.")

    while np.linalg.norm(pos_x_delta) > tol:
        pos_rx_m = np.tile(rx_est_m[0:3,:].T, (num_svs, 1))

        gt_r_m = np.linalg.norm(pos_rx_m - pos_sv_m, axis = 1,
                                 keepdims = True)

        if only_bias:
            geometry_matrix = np.ones((num_svs,1))
        else:
            geometry_matrix = np.ones((num_svs,4))
            geometry_matrix[:,:3] = np.divide(pos_rx_m - pos_sv_m,
                                              gt_r_m.reshape(-1,1))

        # assumes the use of corrected pseudoranges with the satellite
        # clock bias already removed
        pr_delta = corr_pr_m - gt_r_m - rx_est_m[3,0]
        try:
            pos_x_delta = np.linalg.pinv(geometry_matrix.T @ weight_matrix @ geometry_matrix) \
                        @ geometry_matrix.T @ weight_matrix @ pr_delta
        except np.linalg.LinAlgError as exception:
            print(exception)
            break

        if only_bias:
            rx_est_m[3,0] += pos_x_delta[0,0]
        else:
            rx_est_m += pos_x_delta

        count += 1

        if count >= max_count:
            warnings.warn("Newton Raphson did not converge.", RuntimeWarning)
            break

    return rx_est_m
