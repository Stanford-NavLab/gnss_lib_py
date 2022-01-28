"""Visualization functions of GNSS data.

"""

__authors__ = "D. Knowles"
__date__ = "27 Jan 2022"

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.collections import LineCollection

from gnss_lib_py.core.coordinates import LocalCoord

def plot_skyplot(measurements, states):
    """Skyplot of data

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

    root_path = os.path.dirname(
                os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__))))

    skyplot_data = {}
    signal_types = measurements.get_strings("signal_type")
    sv_ids = measurements.get_strings("sv_id")

    pos_sv_m = np.hstack((measurements["x_sat_m",:].reshape(-1,1),
                          measurements["y_sat_m",:].reshape(-1,1),
                          measurements["z_sat_m",:].reshape(-1,1)))

    for ii, timestep in enumerate(np.unique(measurements["millisSinceGpsEpoch",:])):
        idxs = np.where(measurements["millisSinceGpsEpoch",:] == timestep)[1]
        for jj in idxs:

            if signal_types[jj] not in skyplot_data:
                if signal_types[jj] == "GPS_L5" or signal_types[jj] == "GAL_E5A":
                    continue
                skyplot_data[signal_types[jj]] = {}

            if jj == 0:
                lc = LocalCoord.from_ecef(states[0:3,ii])
            sv_ned = lc.ecef2ned(pos_sv_m[jj:jj+1,:])[0]

            sv_az = np.pi/2.-np.arctan2(sv_ned[0],sv_ned[1])
            xy = np.sqrt(sv_ned[0]**2+sv_ned[1]**2)
            sv_el = np.degrees(np.arctan2(-sv_ned[2],xy))

            if sv_ids[jj] not in skyplot_data[signal_types[jj]]:
                skyplot_data[signal_types[jj]][sv_ids[jj]] = [[sv_az],[sv_el]]
            else:
                skyplot_data[signal_types[jj]][sv_ids[jj]][0].append(sv_az)
                skyplot_data[signal_types[jj]][sv_ids[jj]][1].append(sv_el)

    ####################################################################
    # BROKEN UP BY CONSTELLATION TYPE
    ####################################################################


    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='polar')
    cc = 0
    for signal_type, signal_data in skyplot_data.items():
        ss = 0
        color = "C" + str(cc % 10)
        if signal_type == "GPS_L1":
            cmap = "Reds"
            color = "r"
            marker = "o"
        elif signal_type == "GAL_E1":
            cmap = "Blues"
            color = "b"
            marker = "*"
        elif signal_type == "GLO_G1":
            cmap = "Greens"
            color = "g"
            marker = "P"
        for sv, sv_data in signal_data.items():
            points = np.array([sv_data[0], sv_data[1]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(0,len(segments))
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(range(len(segments)))
            lc.set_linewidth(2)
            lin = ax.add_collection(lc)
            if ss == 0:
                # ax.plot(sv_data[0],sv_data[1],c=color,label=signal_type)
                ax.plot(sv_data[0][-1],sv_data[1][-1],c=color,
                        marker=marker, label=signal_type)
            else:
                ax.plot(sv_data[0][-1],sv_data[1][-1],c=color,
                        marker=marker)
            ss += 1
        cc += 1

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_yticks(range(0, 90+10, 30))                   # Define the yticks
    ax.set_ylim(90,0)
    # yLabel = ['90', '', '', '60', '', '', '30', '', '', '']

    ax.legend(bbox_to_anchor=(1.05, 1))

    plt_file = os.path.join(root_path,"dev","skyplot.png")

    fig.savefig(plt_file,
            dpi=300.,
            format="png",
            bbox_inches="tight")

    # close previous figure
    plt.close(fig)

def plot_metric(measurements, metric):
    """Skyplot of data

    Parameters
    ----------
    measurements : gnss_lib_py.parsers.measurement
        Instance of the Measurement class
    metric : string
        Column name for metric to be plotted

    """

    root_path = os.path.dirname(
                os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__))))

    data = {}
    signal_types = measurements.get_strings("signal_type")
    sv_ids = measurements.get_strings("sv_id")

    time0 = measurements["millisSinceGpsEpoch",0]/1000.

    for ii in range(measurements.shape[1]):
        if signal_types[ii] not in data:
            data[signal_types[ii]] = {}
        if sv_ids[ii] not in data[signal_types[ii]]:
            data[signal_types[ii]][sv_ids[ii]] = [[measurements["millisSinceGpsEpoch",ii]/1000. - time0],
                                                  [measurements[metric,ii]]]
        else:
            data[signal_types[ii]][sv_ids[ii]][0].append(measurements["millisSinceGpsEpoch",ii]/1000. - time0)
            data[signal_types[ii]][sv_ids[ii]][1].append(measurements[metric,ii])

    ####################################################################
    # BROKEN UP BY CONSTELLATION TYPE
    ####################################################################

    for signal_type, signal_data in data.items():
        fig = plt.figure(figsize=(5,3))
        ax = plt.gca()
        plt.title(signal_type)

        for sv, sv_data in signal_data.items():
            ax.scatter(sv_data[0],sv_data[1],label=sv,s=5.)

        ax = plt.gca()
        ax.ticklabel_format(useOffset=False)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        plt.xlabel("time [s]")
        plt.ylabel(metric)
        plt.legend(bbox_to_anchor=(1.05, 1))

        plt_file = os.path.join(root_path,"dev", signal_type + "-" + metric + ".png")

        fig.savefig(plt_file,
                dpi=300.,
                format="png",
                bbox_inches="tight")

        # close previous figure
        plt.close(fig)
