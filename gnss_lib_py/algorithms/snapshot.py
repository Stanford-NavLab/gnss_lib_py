"""Point solution methods using GNSS measurements.

This module contains point solution methods for estimating position
at a single GNSS measurement epoch. Position is solved using the
Weighted Least Squares algorithm.

Notes
-----
    Weighted Least Squares solver is not yet implemented. There is not an input
    field for specifying weighting matrix.

"""

__authors__ = "D. Knowles"
__date__ = "25 Jan 2022"

import numpy as np

def solve_wls(measurements):
    """Runs weighted least squares across each timestep.

    Runs weighted least squares across each timestep and adds a new
    columns for the receiver's position and clock bias

    Parameters
    ----------
    measurements : gnss_lib_py.parsers.measurement
        Instance of the Measurement class

    Returns
    -------
    states : np.ndarray
        Estimated receiver position in ECEF frame in meters and the
        estimated receiver clock bias also in meters in an
        array with shape (4 x # timesteps) and the following order of
        rows: x_rx_m, y_rx_m, z_rx_m, b_rx_m.

    """

    if "x_sat_m" not in measurements.rows:
        raise KeyError("x_sat_m (ECEF x position of sv) missing.")
    if "y_sat_m" not in measurements.rows:
        raise KeyError("y_sat_m (ECEF y position of sv) missing.")
    if "z_sat_m" not in measurements.rows:
        raise KeyError("z_sat_m (ECEF z position of sv) missing.")
    if "b_sat_m" not in measurements.rows:
        raise KeyError("b_sat_m (clock bias of sv) missing.")

    for ii, timestep in enumerate(np.unique(measurements["millisSinceGpsEpoch",:])):
        # TODO: make this work across for gps_tow + gps_week
        idxs = np.where(measurements["millisSinceGpsEpoch",:] == timestep)[1]
        pos_sv_m = np.hstack((measurements["x_sat_m",idxs].reshape(-1,1),
                              measurements["y_sat_m",idxs].reshape(-1,1),
                              measurements["z_sat_m",idxs].reshape(-1,1)))
        corr_pr_m = measurements["corr_pr_m",idxs].reshape(-1,1)
        position = wls(np.zeros((4,1)), pos_sv_m, corr_pr_m)
        if ii == 0:
            states = position
        else:
            states = np.hstack((states,position))

    return states

def wls(rx_est_m, pos_sv_m, corr_pr_m, weights = None,
        stationary = False, tol = 1e-7, max_count = 20):
    """Weighted least squares solver for GNSS measurements

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
    stationary : bool
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

    count = 0
    num_svs = pos_sv_m.shape[0]
    if num_svs < 4:
        raise RuntimeError("Need at least four satellites for WLS.")
    pos_x_delta = np.inf*np.ones((4,1))

    while np.max(pos_x_delta) > tol:
        pos_rx_m = np.tile(rx_est_m[0:3,:].T, (num_svs, 1))

        gt_pr_m = np.linalg.norm(pos_rx_m - pos_sv_m, axis = 1,
                                 keepdims = True)

        if stationary:
            geometry_matrix = np.ones((num_svs,1))
        else:
            geometry_matrix = np.ones((num_svs,4))
            geometry_matrix[:,:3] = np.divide(pos_rx_m - pos_sv_m,
                                              gt_pr_m.reshape(-1,1))

        if weights is None:
            weight_matrix = np.eye(num_svs)
        elif isinstance(weights,list):
            raise NotImplementedError("weights not yet implemented in wls")
        else:
            raise TypeError("Weights must be None or list")

        pr_delta = corr_pr_m - gt_pr_m - rx_est_m[3,0]

        pos_x_delta = np.linalg.pinv(weight_matrix @ geometry_matrix) \
                    @ weight_matrix @ pr_delta

        if stationary:
            rx_est_m[3,0] += pos_x_delta[0,0]
        else:
            rx_est_m += pos_x_delta

        count += 1

        if count >= max_count:
            raise RuntimeWarning("Newton Raphson did not converge.")

    return rx_est_m
