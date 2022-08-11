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

    if "x_sv_m" not in measurements.rows:
        raise KeyError("x_sv_m (ECEF x position of sv) missing.")
    if "y_sv_m" not in measurements.rows:
        raise KeyError("y_sv_m (ECEF y position of sv) missing.")
    if "z_sv_m" not in measurements.rows:
        raise KeyError("z_sv_m (ECEF z position of sv) missing.")
    if "b_sv_m" not in measurements.rows:
        raise KeyError("b_sv_m (clock bias of sv) missing.")

    unique_timesteps = np.unique(measurements["gps_millis",:])

    states = np.nan*np.ones((4,len(unique_timesteps)))

    for t_idx, timestep in enumerate(unique_timesteps):
        idxs = np.where(measurements["gps_millis",:] == timestep)[0]
        pos_sv_m = np.hstack((measurements["x_sv_m",idxs].reshape(-1,1),
                              measurements["y_sv_m",idxs].reshape(-1,1),
                              measurements["z_sv_m",idxs].reshape(-1,1)))
        corr_pr_m = measurements["corr_pr_m",idxs].reshape(-1,1)
        if weight_type is not None:
            if isinstance(weight_type,str) and weight_type in measurements.rows:
                weights = measurements[weight_type, idxs].reshape(-1,1)
            else:
                raise TypeError("WLS weights must be None or row"\
                                +" in NavData")
        else:
            weights = None

        position = wls(np.zeros((4,1)), pos_sv_m, corr_pr_m, weights,
                       only_bias, tol, max_count)

        states[:,t_idx:t_idx+1] = position

    state_estimate = NavData()
    state_estimate["x_rx_m"] = states[0,:]
    state_estimate["y_rx_m"] = states[1,:]
    state_estimate["z_rx_m"] = states[2,:]
    state_estimate["b_rx_m"] = states[3,:]

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


        pr_delta = corr_pr_m - gt_r_m - rx_est_m[3,0]
        pos_x_delta = np.linalg.pinv(geometry_matrix.T @ weight_matrix @ geometry_matrix) \
                    @ geometry_matrix.T @ weight_matrix @ pr_delta

        if only_bias:
            rx_est_m[3,0] += pos_x_delta[0,0]
        else:
            rx_est_m += pos_x_delta

        count += 1

        if count >= max_count:
            warnings.warn("Newton Raphson did not converge.", RuntimeWarning)
            break

    return rx_est_m
