"""Visualization functions of GNSS data.

"""

__authors__ = "D. Knowles"
__date__ = "27 Jan 2022"

import os

import numpy as np
import matplotlib as mpl
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb, ListedColormap

import gnss_lib_py.utils.file_operations as fo
from gnss_lib_py.core.coordinates import LocalCoord

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
                   ]
MARKERS = ["o","*","P","v","s","^","p","<","h",">","H","X","D"]

mpl.rcParams['axes.prop_cycle'] = (cycler(color=STANFORD_COLORS) \
                                + cycler(marker=MARKERS))

TIMESTAMP = fo.get_timestamp()

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
    num_vals = 256
    vals = np.ones((num_vals, 4))

    vals[:, 0] = np.linspace(1., rgb_color[0], num_vals)
    vals[:, 1] = np.linspace(1., rgb_color[1], num_vals)
    vals[:, 2] = np.linspace(1., rgb_color[2], num_vals)
    cmap = ListedColormap(vals)

    return cmap

def plot_metric(measurements, metric, save=True, prefix=""):
    """Plot specific metric from a row of the Measurement class.

    Parameters
    ----------
    measurements : gnss_lib_py.parsers.measurement.Measurement
        Instance of the Measurement class
    metric : string
        Row name for metric to be plotted
    save : bool
        Save figure if true, otherwise returns figure object. Defaults
        to saving the figure in the Results folder.
    prefix : string
        File prefix to add to filename.

    Returns
    -------
    figs : list
        List of matplotlib.pyplot.figure objects of residuels, returns
        None if save set to True.

    """

    if len(measurements.str_map[metric]):
        raise KeyError(metric + " is a non-numeric row, unable to plot.")
    if not isinstance(prefix, str):
        raise TypeError("Prefix must be a string.")

    if save: # pragma: no cover
        root_path = os.path.dirname(
                    os.path.dirname(
                    os.path.dirname(
                    os.path.realpath(__file__))))
        log_path = os.path.join(root_path,"results",TIMESTAMP)
        fo.mkdir(log_path)
    else:
        figs = []

    data = {}
    signal_types = measurements.get_strings("signal_type")
    sv_ids = measurements.get_strings("sv_id")

    time0 = measurements["millisSinceGpsEpoch",0]/1000.

    for m_idx in range(measurements.shape[1]):
        if signal_types[m_idx] not in data:
            data[signal_types[m_idx]] = {}
        if sv_ids[m_idx] not in data[signal_types[m_idx]]:
            data[signal_types[m_idx]][sv_ids[m_idx]] = [[measurements["millisSinceGpsEpoch",m_idx]/1000. - time0],
                                                  [measurements[metric,m_idx]]]
        else:
            data[signal_types[m_idx]][sv_ids[m_idx]][0].append(measurements["millisSinceGpsEpoch",m_idx]/1000. - time0)
            data[signal_types[m_idx]][sv_ids[m_idx]][1].append(measurements[metric,m_idx])

    ####################################################################
    # BROKEN UP BY CONSTELLATION TYPE
    ####################################################################

    for signal_type, signal_data in data.items():
        fig = plt.figure(figsize=(5,3))
        axes = plt.gca()
        plt.title(signal_type)

        for sv_name, sv_data in signal_data.items():
            axes.scatter(sv_data[0],sv_data[1],label=sv_name,s=5.)

        axes = plt.gca()
        axes.ticklabel_format(useOffset=False)
        axes.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        plt.xlabel("time [s]")
        plt.ylabel(metric)
        plt.legend(bbox_to_anchor=(1.05, 1))

        if save: # pragma: no cover
            plt_file = os.path.join(log_path, prefix + "_" + metric \
                     + "_" + signal_type + ".png")

            fo.save_figure(fig, plt_file)

            # close previous figure
            plt.close(fig)

        else:
            figs.append(fig)

    if save: # pragma: no cover
        return None
    return figs

def plot_skyplot(measurements, state_estimate, save=True, prefix=""):
    """Skyplot of data

    Parameters
    ----------
    measurements : gnss_lib_py.parsers.measurement.Measurement
        Instance of the Measurement class
    state_estimate : gnss_lib_py.parsers.measurement.Measurement
        Estimated receiver position in ECEF frame in meters and the
        estimated receiver clock bias also in meters as an instance of
        the Measurement class with shape (4 x # unique timesteps) and
        the following rows: x_rx_m, y_rx_m, z_rx_m, b_rx_m.
    save : bool
        Save figure if true, otherwise returns figure object. Defaults
        to saving the figure in the Results folder.
    prefix : string
        File prefix to add to filename.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Figure object of skyplot, returns None if save set to True.

    """

    if not isinstance(prefix, str):
        raise TypeError("Prefix must be a string.")

    local_coord = None

    skyplot_data = {}
    signal_types = list(measurements.get_strings("signal_type"))
    sv_ids = measurements.get_strings("sv_id")

    pos_sv_m = np.hstack((measurements["x_sv_m",:].reshape(-1,1),
                          measurements["y_sv_m",:].reshape(-1,1),
                          measurements["z_sv_m",:].reshape(-1,1)))

    for t_idx, timestep in enumerate(np.unique(measurements["millisSinceGpsEpoch",:])):
        idxs = np.where(measurements["millisSinceGpsEpoch",:] == timestep)[1]
        for m_idx in idxs:

            if signal_types[m_idx] not in skyplot_data:
                if "5" in signal_types[m_idx]:
                    continue
                skyplot_data[signal_types[m_idx]] = {}

            if local_coord is None:
                local_coord = LocalCoord.from_ecef(state_estimate[["x_rx_m","y_rx_m","z_rx_m"],t_idx])
            sv_ned = local_coord.ecef2ned(pos_sv_m[m_idx:m_idx+1,:])[0]

            sv_az = np.pi/2.-np.arctan2(sv_ned[0],sv_ned[1])
            xy_dist = np.sqrt(sv_ned[0]**2+sv_ned[1]**2)
            sv_el = np.degrees(np.arctan2(-sv_ned[2],xy_dist))

            if sv_ids[m_idx] not in skyplot_data[signal_types[m_idx]]:
                skyplot_data[signal_types[m_idx]][sv_ids[m_idx]] = [[sv_az],[sv_el]]
            else:
                skyplot_data[signal_types[m_idx]][sv_ids[m_idx]][0].append(sv_az)
                skyplot_data[signal_types[m_idx]][sv_ids[m_idx]][1].append(sv_el)

    ####################################################################
    # BROKEN UP BY CONSTELLATION TYPE
    ####################################################################


    fig = plt.figure(figsize=(5,5))
    axes = fig.add_subplot(111, projection='polar')
    c_idx = 0
    for signal_type, signal_data in skyplot_data.items():
        s_idx = 0
        color = "C" + str(c_idx % len(STANFORD_COLORS))
        cmap = new_cmap(to_rgb(color))
        marker = MARKERS[c_idx % len(MARKERS)]
        for _, sv_data in signal_data.items():
            # only plot ~ 50 points for each sat to decrease time
            # it takes to plot these line collections
            step = max(1,int(len(sv_data[0])/50.))
            points = np.array([sv_data[0][::step],
                               sv_data[1][::step]]).T
            points = np.reshape(points,(-1, 1, 2))
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(0,len(segments))
            local_coord = LineCollection(segments, cmap=cmap, norm=norm,
                                array = range(len(segments)),
                                linewidths=(4,))
            axes.add_collection(local_coord)
            if s_idx == 0:
                # axes.plot(sv_data[0],sv_data[1],c=color,label=signal_type)
                axes.plot(sv_data[0][-1],sv_data[1][-1],c=color,
                        marker=marker, markersize=8,
                        label=signal_type.replace("_"," "))
            else:
                axes.plot(sv_data[0][-1],sv_data[1][-1],c=color,
                        marker=marker, markersize=8)
            s_idx += 1
        c_idx += 1

    axes.set_theta_zero_location('N')
    axes.set_theta_direction(-1)
    axes.set_yticks(range(0, 90+10, 30))                   # Define the yticks
    axes.set_ylim(90,0)

    axes.legend(bbox_to_anchor=(1.05, 1))

    if save: # pragma: no cover
        root_path = os.path.dirname(
                    os.path.dirname(
                    os.path.dirname(
                    os.path.realpath(__file__))))
        log_path = os.path.join(root_path,"results",TIMESTAMP)
        fo.mkdir(log_path)
        plt_file = os.path.join(log_path,prefix+"_skyplot.png")

        fo.save_figure(fig, plt_file)

        # close previous figure
        plt.close(fig)

        return None

    return fig


def plot_skyplot_from_status(status, save=True, prefix=""):
    """Skyplot of data

    Parameters
    ----------
    status :
    save : bool
        Save figure if true, otherwise returns figure object. Defaults
        to saving the figure in the Results folder.
    prefix : string
        File prefix to add to filename.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Figure object of skyplot, returns None if save set to True.

    """

    if not isinstance(prefix, str):
        raise TypeError("Prefix must be a string.")

    skyplot_data = {}

    for m_idx, row in status.iterrows():

        if row["ConstellationType"] not in skyplot_data:
            # if "5" in signal_types[m_idx]:
            #     continue
            skyplot_data[row["ConstellationType"]] = {}

        sv_az = np.radians(row["AzimuthDegrees"])
        sv_el = row["ElevationDegrees"]

        if row["Svid"] not in skyplot_data[row["ConstellationType"]]:
            skyplot_data[row["ConstellationType"]][row["Svid"]] = [[sv_az],[sv_el]]
        else:
            skyplot_data[row["ConstellationType"]][row["Svid"]][0].append(sv_az)
            skyplot_data[row["ConstellationType"]][row["Svid"]][1].append(sv_el)

    ####################################################################
    # BROKEN UP BY CONSTELLATION TYPE
    ####################################################################


    fig = plt.figure(figsize=(5,5))
    axes = fig.add_subplot(111, projection='polar')
    c_idx = 0
    for signal_type, signal_data in skyplot_data.items():
        s_idx = 0
        color = "C" + str(c_idx % len(STANFORD_COLORS))
        cmap = new_cmap(to_rgb(color))
        marker = MARKERS[c_idx % len(MARKERS)]
        for _, sv_data in signal_data.items():
            # only plot ~ 50 points for each sat to decrease time
            # it takes to plot these line collections
            step = max(1,int(len(sv_data[0])/50.))
            points = np.array([sv_data[0][::step],
                               sv_data[1][::step]]).T
            points = np.reshape(points,(-1, 1, 2))
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(0,len(segments))
            local_coord = LineCollection(segments, cmap=cmap, norm=norm,
                                array = range(len(segments)),
                                linewidths=(4,))
            axes.add_collection(local_coord)
            if s_idx == 0:
                # axes.plot(sv_data[0],sv_data[1],c=color,label=signal_type)
                axes.plot(sv_data[0][-1],sv_data[1][-1],c=color,
                        marker=marker, markersize=8,
                        label=signal_type)
            else:
                axes.plot(sv_data[0][-1],sv_data[1][-1],c=color,
                        marker=marker, markersize=8)
            s_idx += 1
        c_idx += 1

    axes.set_theta_zero_location('N')
    axes.set_theta_direction(-1)
    axes.set_yticks(range(0, 90+10, 30))                   # Define the yticks
    axes.set_ylim(90,0)

    axes.legend(bbox_to_anchor=(1.05, 1))

    if save: # pragma: no cover
        root_path = os.path.dirname(
                    os.path.dirname(
                    os.path.dirname(
                    os.path.realpath(__file__))))
        log_path = os.path.join(root_path,"results",TIMESTAMP)
        fo.mkdir(log_path)
        plt_file = os.path.join(log_path,prefix+"_skyplot.png")

        fo.save_figure(fig, plt_file)

        # close previous figure
        plt.close(fig)

        return None

    return fig


def plot_residuals(measurements, save=True, prefix=""):
    """Plot residuals.

    Parameters
    ----------
    measurements : gnss_lib_py.parsers.measurement.Measurement
        Instance of the Measurement class
    save : bool
        Save figure if true, otherwise returns figure object. Defaults
        to saving the figure in the Results folder.
    prefix : string
        File prefix to add to filename.

    Returns
    -------
    figs : list
        List of matplotlib.pyplot.figure objects of residuels, returns
        None if save set to True.

    """

    if "residuals" not in measurements.rows:
        raise KeyError("residuals missing, run solve_residuals().")
    if not isinstance(prefix, str):
        raise TypeError("Prefix must be a string.")

    if save: # pragma: no cover
        root_path = os.path.dirname(
                    os.path.dirname(
                    os.path.dirname(
                    os.path.realpath(__file__))))
        log_path = os.path.join(root_path,"results",TIMESTAMP)
        fo.mkdir(log_path)
    else:
        figs = []

    residual_data = {}
    signal_types = measurements.get_strings("signal_type")
    sv_ids = measurements.get_strings("sv_id")

    time0 = measurements["millisSinceGpsEpoch",0]/1000.

    for m_idx in range(measurements.shape[1]):
        if signal_types[m_idx] not in residual_data:
            residual_data[signal_types[m_idx]] = {}
        if sv_ids[m_idx] not in residual_data[signal_types[m_idx]]:
            residual_data[signal_types[m_idx]][sv_ids[m_idx]] = [[measurements["millisSinceGpsEpoch",m_idx]/1000. - time0],
                        [measurements["residuals",m_idx]]]
        else:
            residual_data[signal_types[m_idx]][sv_ids[m_idx]][0].append(measurements["millisSinceGpsEpoch",m_idx]/1000. - time0)
            residual_data[signal_types[m_idx]][sv_ids[m_idx]][1].append(measurements["residuals",m_idx])

    ####################################################################
    # BROKEN UP BY CONSTELLATION TYPE
    ####################################################################


    for signal_type, signal_residuals in residual_data.items():
        fig = plt.figure(figsize=(5,3))

        plt.title(signal_type.replace("_"," "))
        signal_type_svs = list(signal_residuals.keys())

        for sv_name, sv_data in signal_residuals.items():
            color = STANFORD_COLORS[signal_type_svs.index(sv_name)]
            plt.plot(sv_data[0], sv_data[1],
                    color = color,
                    label = signal_type.replace("_"," ") + " " + str(sv_name))
        axes = plt.gca()
        axes.ticklabel_format(useOffset=False)
        axes.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        plt.ylim(-100.,100.)
        plt.xlabel("time [s]")
        plt.ylabel("residiual [m]")
        plt.legend(bbox_to_anchor=(1.05, 1))

        if save: # pragma: no cover
            plt_file = os.path.join(log_path, prefix + "_residuals_" \
                     + signal_type + ".png")

            fo.save_figure(fig, plt_file)

            # close previous figure
            plt.close(fig)
        else:
            figs.append(fig)

    if save: # pragma: no cover
        return None
    return figs
