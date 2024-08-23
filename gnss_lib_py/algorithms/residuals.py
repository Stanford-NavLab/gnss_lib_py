"""Calculate residuals

"""

__authors__ = "D. Knowles"
__date__ = "25 Jan 2022"

import numpy as np

from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.navdata.operations import loop_time, find_wildcard_indexes

def solve_residuals(measurements, receiver_state, inplace=True):
    """Calculates residuals given pseudoranges and receiver position.

    Parameters
    ----------
    measurements : gnss_lib_py.navdata.navdata.NavData
        Instance of the NavData class which must include ``gps_millis``
        and ``corr_pr_m``.
    receiver_state : gnss_lib_py.navdata.navdata.NavData
        Either estimated or ground truth receiver position in ECEF frame
        in meters and the estimated or ground truth receiver clock bias
        also in meters as an instance of the NavData class with the
        following rows: x_rx*_m, y_rx*_m, z_rx*_m, b_rx*_m.
    inplace : bool
        If False, will return new NavData instance with gps_millis and
        reisuals. If True, will add a "residuals_m" rows in the
        current NavData instance.

    Returns
    -------
    new_navdata : gnss_lib_py.navdata.navdata.NavData or None
        If inplace is False, returns new NavData instance containing
        "gps_millis" and residual rows. If inplace is True, returns
        None.

    """

    # verify corrected pseudoranges exist in inputs
    measurements.in_rows(["gps_millis","corr_pr_m"])
    receiver_state.in_rows(["gps_millis"])


    rx_idxs = find_wildcard_indexes(receiver_state,["x_rx*_m",
                                                    "y_rx*_m",
                                                    "z_rx*_m",
                                                    "b_rx*_m"],
                                                    max_allow=1)

    residuals = []
    for timestamp, _, measurement_subset in loop_time(measurements,"gps_millis"):

        pos_sv_m = measurement_subset[["x_sv_m","y_sv_m","z_sv_m"]].T
        pos_sv_m = np.atleast_2d(pos_sv_m)

        num_svs = pos_sv_m.shape[0]

        corr_pr_m = measurement_subset["corr_pr_m"].reshape(-1,1)

        # find time index for receiver_state NavData instance
        rx_t_idx = np.argmin(np.abs(receiver_state["gps_millis"] - timestamp))

        rx_pos = receiver_state[[rx_idxs["x_rx*_m"][0],
                                 rx_idxs["y_rx*_m"][0],
                                 rx_idxs["z_rx*_m"][0]],
                                 rx_t_idx].reshape(1,-1)
        pos_rx_m = np.tile(rx_pos, (num_svs, 1))

        # assumes the use of corrected pseudoranges with the satellite
        # clock bias already removed
        gt_pr_m = np.linalg.norm(pos_rx_m - pos_sv_m, axis = 1,
                                 keepdims = True) \
                + receiver_state[rx_idxs["b_rx*_m"][0],rx_t_idx]

        # calculate residual
        residuals_epoch = corr_pr_m - gt_pr_m
        residuals += residuals_epoch.reshape(-1).tolist()

    if inplace:
        # add measurements to measurement class
        measurements["residuals_m"] = residuals
        return None

    # if not inplace, create new NavData instance to return
    residual_navdata = NavData()
    residual_navdata["residuals_m"] = residuals
    residual_navdata["gps_millis"]= measurements["gps_millis"]
    for row in ["gnss_id","sv_id","signal_type"]:
        if row in measurements.rows:
            residual_navdata[row] = measurements[row]

    return residual_navdata
