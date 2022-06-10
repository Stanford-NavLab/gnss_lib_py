"""Visualization functions of GNSS data.

"""

__authors__ = "D. Knowles"
__date__ = "27 Jan 2022"

import os

import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb, ListedColormap

from gnss_lib_py.core.coordinates import LocalCoord

# TODO:  make into fancy class or matplotlib color to automatically loop?
STANFORD_COLORS = [
                   "#8C1515",   # cardinal red
                   "#6FC3FF",   # light digital blue
                   "#006F54",   # dark digital green
                   "#620059",   # plum
                   "#E98300",   # poppy
                   "#FEDD5C",   # illuminating
                   "#E04F39",   # spirited
                   "#4298B5",   # sky
                   "#8F993E",   # olive
                   "#651C32",   # brick
                   "#B1040E",   # digital red
                   "#016895",   # dark sky
                   "#279989",   # palo verde
                   # "#67AFD2",   # light sky
                   # "#008566",   # digital green
                   # "",   #
                   # "",   #
                   # "",   #
                   # "",   #
                   ]

def new_cmap(rgb_color):
    """Return a new cmap from a color going to white.

    Given an RGB color, it creates a new color map that starts at white
    then fades into the provided RGB color.

    Parameters
    ----------
    rgb_color : tuple
        color tuple of (red, green, blue) in floats between 0 and 1.0

    Returns
    -------
    cmap : ListedColormap
        New color map made from the provided color.


    Notes
    -----
    More details and examples at the following link
    https://matplotlib.org/3.1.0/tutorials/colors/colormap-manipulation.html

    """
    N = 256
    vals = np.ones((N, 4))

    vals[:, 0] = np.linspace(1., rgb_color[0], N)
    vals[:, 1] = np.linspace(1., rgb_color[1], N)
    vals[:, 2] = np.linspace(1., rgb_color[2], N)
    cmap = ListedColormap(vals)

    return cmap

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
    signal_types = list(measurements.get_strings("signal_type"))
    sv_ids = measurements.get_strings("sv_id")

    pos_sv_m = np.hstack((measurements["x_sv_m",:].reshape(-1,1),
                          measurements["y_sv_m",:].reshape(-1,1),
                          measurements["z_sv_m",:].reshape(-1,1)))

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
            color = to_rgb(STANFORD_COLORS[signal_types.index("GPS_L1")])
            cmap = new_cmap(color)
            marker = "o"
        elif signal_type == "GAL_E1":
            color = to_rgb(STANFORD_COLORS[signal_types.index("GAL_E1")])
            cmap = new_cmap(color)
            marker = "*"
        elif signal_type == "GLO_G1":
            color = to_rgb(STANFORD_COLORS[signal_types.index("GLO_G1")])
            cmap = new_cmap(color)
            marker = "P"
        for sv, sv_data in signal_data.items():
            # only plot ~ 50 points for each sat to decrease time
            # it takes to plot these line collections
            step = max(1,int(len(sv_data[0])/50.))
            points = np.array([sv_data[0][::step],
                               sv_data[1][::step]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(0,len(segments))
            lc = LineCollection(segments, cmap=cmap, norm=norm,
                                array = range(len(segments)),
                                linewidths=(4,))
            lin = ax.add_collection(lc)
            if ss == 0:
                # ax.plot(sv_data[0],sv_data[1],c=color,label=signal_type)
                ax.plot(sv_data[0][-1],sv_data[1][-1],c=color,
                        marker=marker, markersize=12,
                        label=signal_type.replace("_"," "))
            else:
                ax.plot(sv_data[0][-1],sv_data[1][-1],c=color,
                        marker=marker, markersize=12)
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

        plt.title(signal_type.replace("_"," "))
        signal_type_svs = list(signal_residuals.keys())

        for sv, sv_data in signal_residuals.items():
            color = STANFORD_COLORS[signal_type_svs.index(sv)]
            plt.plot(sv_data[0], sv_data[1],
                    color = color,
                    label = signal_type.split("_")[0] + " " + sv)
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


def visualize_traffic(df, center, zoom=9):
    fig = px.scatter_mapbox(df,

                            # Here, plotly gets, (x,y) coordinates
                            lat="latDeg",
                            lon="lngDeg",

                            #Here, plotly detects color of series
                            color='Label',
                            labels='Label',

                            zoom=zoom,
                            center=center,
                            height=600,
                            width=800)
    fig.update_layout(mapbox_style='stamen-terrain')
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
#     fig.update_layout(title_text="Ground Truth Tracks of Android Smartphone GNSS Dataset")
#     fig.legend()
    fig.show()
