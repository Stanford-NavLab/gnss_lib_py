"""Calculate residuals

"""

__authors__ = "D. Knowles"
__date__ = "25 Jan 2022"

import numpy as np

def solve_residuals(measurements, state_estimate):
    """Calculates residuals

    Parameters
    ----------
    measurements : gnss_lib_py.parsers.navdata.NavData
        Instance of the NavData class
    state_estimate : gnss_lib_py.parsers.navdata.NavData
        Estimated receiver position in ECEF frame in meters and the
        estimated receiver clock bias also in meters as an instance of
        the NavData class with shape (4 x # unique timesteps) and
        the following rows: x_rx_m, y_rx_m, z_rx_m, b_rx_m.

    """


    residuals = np.nan*np.ones((1,len(measurements)))

    unique_timesteps = np.unique(measurements["gps_millis",:])
    for t_idx, timestep in enumerate(unique_timesteps):
        idxs = np.where(measurements["gps_millis",:] == timestep)[0]

        pos_sv_m = np.hstack((measurements["x_sv_m",idxs].reshape(-1,1),
                              measurements["y_sv_m",idxs].reshape(-1,1),
                              measurements["z_sv_m",idxs].reshape(-1,1)))

        num_svs = pos_sv_m.shape[0]

        corr_pr_m = measurements["corr_pr_m",idxs].reshape(-1,1)


        rx_pos = state_estimate[["x_rx_m","y_rx_m","z_rx_m"],t_idx:t_idx+1]
        pos_rx_m = np.tile(rx_pos.T, (num_svs, 1))

        gt_pr_m = np.linalg.norm(pos_rx_m - pos_sv_m, axis = 1,
                                 keepdims = True) \
                + state_estimate["b_rx_m",t_idx]

        # calculate residual
        residuals_epoch = corr_pr_m - gt_pr_m

        residuals[:,idxs] = residuals_epoch.T

    # add measurements to measurement class
    measurements["residuals"] = residuals
