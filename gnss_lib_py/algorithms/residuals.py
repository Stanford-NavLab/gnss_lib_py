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
    measurements : gnss_lib_py.parsers.measurement
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

        pos_sv_m = np.hstack((measurements["x_sat_m",idxs].reshape(-1,1),
                              measurements["y_sat_m",idxs].reshape(-1,1),
                              measurements["z_sat_m",idxs].reshape(-1,1)))

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


def plot_residuals(measurements):
    """Plot residuals nicely

    Parameters
    ----------
    measurements : gnss_lib_py.parsers.measurement
        Instance of the Measurement class

    """

    root_path = os.path.dirname(
                os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__))))

    residual_data = {}
    signal_types = measurements.get_strings("signal_type")
    sv_ids = measurements.get_strings("sv_id")

    time0 = measurements["millisSinceGpsEpoch",0]/1000.

    for ii in range(measurements.shape[1]):
        if signal_types[ii] not in residual_data:
            residual_data[signal_types[ii]] = {}
        if sv_ids[ii] not in residual_data[signal_types[ii]]:
            residual_data[signal_types[ii]][sv_ids[ii]] = [[measurements["millisSinceGpsEpoch",ii]/1000. - time0],
                        [measurements["residuals",ii]]]
        else:
            residual_data[signal_types[ii]][sv_ids[ii]][0].append(measurements["millisSinceGpsEpoch",ii]/1000. - time0)
            residual_data[signal_types[ii]][sv_ids[ii]][1].append(measurements["residuals",ii])

    ####################################################################
    # BROKEN UP BY CONSTELLATION TYPE
    ####################################################################


    for signal_type, signal_residuals in residual_data.items():
        fig = plt.figure(figsize=(5,3))

        plt.title(signal_type)

        for sv, sv_data in signal_residuals.items():
            plt.plot(sv_data[0],sv_data[1],label=sv)
        ax = plt.gca()
        ax.ticklabel_format(useOffset=False)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        plt.ylim(-100.,100.)
        plt.xlabel("time [s]")
        plt.ylabel("residiual [m]")
        plt.legend(bbox_to_anchor=(1.05, 1))

        plt_file = os.path.join(root_path,"dev", signal_type + "-residuals.png")

        fig.savefig(plt_file,
                dpi=300.,
                format="png",
                bbox_inches="tight")

        # close previous figure
        plt.close(fig)
