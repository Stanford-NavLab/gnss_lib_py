"""Point solution methods using GNSS measurements.

This module contains point solution methods for estimating position
at a single GNSS measurement epoch. Position is solved using the
Weighted Least Squares algorithm.

"""

__authors__ = "D. Knowles, A. Kanhere, Shubh Gupta, Bradley Collicott"
__date__ = "25 Jan 2022"

import warnings

import numpy as np

from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.utils import constants as consts
from gnss_lib_py.navdata.operations import loop_time, find_wildcard_indexes
from gnss_lib_py.utils.coordinates import ecef_to_geodetic

def solve_wls(measurements, weight_type = None, only_bias = False,
              receiver_state=None, tol = 1e-7, max_count = 20,
              sv_rx_time=False, delta_t_decimals=-2):
    """Runs weighted least squares across each timestep.

    Runs weighted least squares across each timestep and adds a new
    row for the receiver's position and clock bias.

    The option for only_bias allows the user to only calculate the clock
    bias if the receiver position is already known. Only the bias term
    in b_rx_wls_m will be updated if only_bias is set to True.

    If only_bias is set to True, then the receiver position must also
    be passed in as the receiver_state.

    Parameters
    ----------
    measurements : gnss_lib_py.navdata.navdata.NavData
        Instance of the NavData class which must include at least
        ``gps_millis``, ``x_sv_m``, ``y_sv_m``, and ``z_sv_m``
    weight_type : string
        Must either be None or the name of a row in measurements
    only_bias : bool
        If True, then only the receiver clock bias is estimated.
        Otherwise, both position and clock bias are estimated.
    receiver_state : gnss_lib_py.navdata.navdata.NavData
        Only used if only_bias is set to True, see description above.
        Receiver position in ECEF frame in meters as an instance of the
        NavData class with at least the following rows: ``x_rx*_m``,
        ``y_rx*_m``, ``z_rx*_m``, ``gps_millis``.
    tol : float
        Tolerance used for the convergence check.
    max_count : int
        Number of maximum iterations before process is aborted and
        solution returned.
    sv_rx_time : bool
        Flag that specifies whether the input SV positions are in the ECEF
        frame of reference corresponding to when the measurements were
        received. If set to `True`, the satellite positions are used as
        is. The default value is `False`, in which case the ECEF positions
        are assumed to in the ECEF frame at the time of signal transmission
        and are converted to the ECEF frame at the time of signal reception,
        while solving the WLS problem.
    delta_t_decimals : int
            Decimal places after which times are considered equal.

    Returns
    -------
    state_estimate : gnss_lib_py.navdata.navdata.NavData
        Estimated receiver position in ECEF frame in meters and the
        estimated receiver clock bias also in meters as an instance of
        the NavData class with shape (4 x # unique timesteps) and
        the following rows: gps_millis, x_rx_wls_m, y_rx_wls_m,
        z_rx_wls_m, b_rx_wls_m, lat_rx_wls_deg, lon_rx_wls_deg,
        alt_rx_wls_m.

    """

    # check that all necessary rows exist
    measurements.in_rows(["x_sv_m","y_sv_m","z_sv_m","gps_millis"])

    if only_bias:
        if receiver_state is None:
            raise RuntimeError("receiver_state needed in WLS " \
                    + "for only_bias.")

        rx_rows_to_find = ['x_rx*_m', 'y_rx*_m', 'z_rx*_m']
        rx_idxs = find_wildcard_indexes(receiver_state,
                                               rx_rows_to_find,
                                               max_allow=1)
    states = []
    runtime_error_idxs = {}

    position = np.zeros((4,1))
    for timestamp, _, measurement_subset in loop_time(measurements,"gps_millis",
                                                      delta_t_decimals=delta_t_decimals):

        pos_sv_m = measurement_subset[["x_sv_m","y_sv_m","z_sv_m"]].T
        pos_sv_m = np.atleast_2d(pos_sv_m)

        corr_pr_m = measurement_subset["corr_pr_m"].reshape(-1,1)

        # remove NaN indexes
        not_nan_indexes = ~np.isnan(pos_sv_m).any(axis=1) & ~np.isnan(corr_pr_m).any(axis=1)
        pos_sv_m = pos_sv_m[not_nan_indexes]
        corr_pr_m = corr_pr_m[not_nan_indexes]

        if weight_type is not None:
            if isinstance(weight_type,str) and weight_type in measurements.rows:
                weights = measurement_subset[weight_type].reshape(-1,1)
                weights = weights[not_nan_indexes]
            else:
                raise TypeError("WLS weights must be None or row"\
                                +" in NavData")
        else:
            weights = None

        try:
            if only_bias:
                position = np.vstack((
                                  receiver_state.where("gps_millis",
                                  timestamp)[[rx_idxs["x_rx*_m"][0],
                                              rx_idxs["y_rx*_m"][0],
                                              rx_idxs["z_rx*_m"][0]]
                                              ,0].reshape(-1,1),
                                             position[3])) # clock bias
            position = wls(position, pos_sv_m, corr_pr_m, weights,
                           only_bias, tol, max_count, sv_rx_time=sv_rx_time)
            states.append([timestamp] + np.squeeze(position).tolist())
        except RuntimeError as error:
            if str(error) not in runtime_error_idxs:
                runtime_error_idxs[str(error)] = [str(int(timestamp))]
            else:
                runtime_error_idxs[str(error)].append(str(int(timestamp)))
            states.append([timestamp, np.nan, np.nan, np.nan, np.nan])

    if len(states) == 0:
        states = np.empty((0,5))
    states = np.array(states)

    state_estimate = NavData()
    state_estimate["gps_millis"] = states[:,0]
    state_estimate["x_rx_wls_m"] = states[:,1]
    state_estimate["y_rx_wls_m"] = states[:,2]
    state_estimate["z_rx_wls_m"] = states[:,3]
    state_estimate["b_rx_wls_m"] = states[:,4]

    for error,timestamps in runtime_error_idxs.items():
        warnings.warn(error + " Encountered at " + str(len(timestamps))\
                    + " gps_millis of: " \
                + ", ".join(timestamps), RuntimeWarning)

    if np.isnan(states[:,1:]).all():
        warnings.warn("No valid state estimate computed in WLS, "\
                    + "returning NaNs.", RuntimeWarning)
        return state_estimate

    lat,lon,alt = ecef_to_geodetic(state_estimate[["x_rx_wls_m","y_rx_wls_m",
                                   "z_rx_wls_m"]].reshape(3,-1))
    state_estimate["lat_rx_wls_deg"] = lat
    state_estimate["lon_rx_wls_deg"] = lon
    state_estimate["alt_rx_wls_m"] = alt

    return state_estimate

def wls(rx_est_m, pos_sv_m, corr_pr_m, weights = None,
        only_bias = False, tol = 1e-7, max_count = 20, sv_rx_time=False):
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
        Satellite ECEF positions as an array of shape [# svs x 3] where
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
    sv_rx_time : bool
        Flag to indicate if the satellite positions at the time of
        transmission should be used as is or if they should be transformed
        to the ECEF frame of reference at the time of reception. For real
        measurements, use ``sv_rx_time=False`` to account for the Earth's
        rotation and convert SV positions from the ECEF frame at the time
        of signal transmission to the ECEF frame at the time of signal
        reception. If the SV positions should be used as is, set
        ``sv_rx_time=True`` to indicate that the given positions are in
        the ECEF frame of reference for when the signals are received.
        By default, ``sv_rx_time=False``.

    Returns
    -------
    rx_est_m : np.ndarray
        Estimated receiver position in ECEF frame in meters and the
        estimated receiver clock bias also in meters in an
        array with shape (4 x 1) and the following order:
        x_rx_m, y_rx_m, z_rx_m, b_rx_m.

    Notes
    -----
    This function internally updates the used SV position to account for
    the time taken for the signal to travel to the Earth from the GNSS
    satellites.
    Since the SV and receiver positions are calculated in an ECEF frame
    of reference, which is moving with the Earth's rotation, the reference
    frame is slightly (about 30 m along longitude) different when the
    signals are received than when the signals were transmitted. Given
    the receiver's position is estimated when the signal is received,
    the SV positions need to be updated to reflect the change in the
    frame of reference in which their position is calculated.

    This update happens after every Gauss-Newton update step and is
    adapted from [1]_.

    References
    ----------
    .. [1] https://github.com/google/gps-measurement-tools/blob/master/opensource/FlightTimeCorrection.m

    """

    rx_est_m = rx_est_m.copy() # don't change referenced value

    count = 0
    # Store the SV position at the original receiver time.
    # This position will be modified by the time taken by the signal to
    # travel to the receiver.
    rx_time_pos_sv_m = pos_sv_m.copy()
    num_svs = pos_sv_m.shape[0]
    if num_svs < 4 and not only_bias:
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

        if not sv_rx_time:
            # Update the satellite positions based on the time taken for
            # the signal to reach the Earth and the satellite clock bias.
            delta_t = (corr_pr_m.reshape(-1) - rx_est_m[3,0])/consts.C
            dtheta = consts.OMEGA_E_DOT*delta_t
            pos_sv_m[:, 0] = np.cos(dtheta)*rx_time_pos_sv_m[:,0] + \
                             np.sin(dtheta)*rx_time_pos_sv_m[:,1]
            pos_sv_m[:, 1] = -np.sin(dtheta)*rx_time_pos_sv_m[:,0] + \
                              np.cos(dtheta)*rx_time_pos_sv_m[:,1]

        count += 1

        if count >= max_count:
            warnings.warn("Newton Raphson did not converge.", RuntimeWarning)
            break

    return rx_est_m
