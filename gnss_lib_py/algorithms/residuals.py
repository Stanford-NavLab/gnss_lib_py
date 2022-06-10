"""Calculate residuals

"""

__authors__ = "D. Knowles"
__date__ = "25 Jan 2022"

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from gnss_lib_py.core.coordinates import LocalCoord

def calc_residuals(measurements, states):
    """Calculates residuals

    Parameters
    ----------
    measurements : gnss_lib_py.parsers.measurement.Measurement
        Instance of the Measurement class
    states : np.ndarray
        Estimated receiver position in ECEF frame in meters and the
        estimated receiver clock bias also in meters in an
        array with shape (4 x # timesteps) and the following order of
        rows: x_rx_m, y_rx_m, z_rx_m, b_rx_m.

    """

    for ii, timestep in enumerate(np.unique(measurements["millisSinceGpsEpoch",:])):
        # TODO: make this work across for gps_tow + gps_week
        idxs = np.where(measurements["millisSinceGpsEpoch",:] == timestep)[1]

        pos_sv_m = np.hstack((measurements["x_sv_m",idxs].reshape(-1,1),
                              measurements["y_sv_m",idxs].reshape(-1,1),
                              measurements["z_sv_m",idxs].reshape(-1,1)))

        num_svs = pos_sv_m.shape[0]
        if num_svs < 4:
            raise RuntimeError("Need at least four satellites for WLS.")

        corr_pr_m = measurements["corr_pr_m",idxs].reshape(-1,1)

        pos_rx_m = np.tile(states[0:3,ii:ii+1].T, (num_svs, 1))

        gt_pr_m = np.linalg.norm(pos_rx_m - pos_sv_m, axis = 1,
                                 keepdims = True)

        # calculate residual
        residuals_epoch = corr_pr_m - gt_pr_m - states[3,ii]

        if ii == 0:
            residuals = residuals_epoch
        else:
            residuals = np.vstack((residuals,residuals_epoch))

    # add measurements to measurement class
    measurements["residuals"] = residuals
