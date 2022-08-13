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
from gnss_lib_py.utils.coordinates import LocalCoord

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

mpl.rcParams['axes.prop_cycle'] = cycler(color=STANFORD_COLORS)

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


def plot_metric(navdata, *args, save=True, prefix=""):
    """Plot specific metric from a row of the NavData class.

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Instance of the NavData class
    *args : tuple
        Tuple of row names that are to be plotted. If one is given, that
        value is plotted on the y-axis. If two values are given, the
        first is plotted on the x-axis and the second on the y-axis.
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
    if len(args)==1:
        x_metric = None
        y_metric = args[0]
    elif len(args)==2:
        x_metric = args[0]
        y_metric = args[1]
    else:
        raise ValueError("Cannot plot more than 1 pair of x-y values")

    if len(navdata.str_map[y_metric]):
        raise KeyError(y_metric + " is a non-numeric row, unable to plot.")
    if x_metric is not None and len(navdata.str_map[x_metric]):
        raise KeyError(x_metric + " is a non-numeric row, unable to plot.")
    if not isinstance(prefix, str):
        raise TypeError("Prefix must be a string.")

    if save: # pragma: no cover
        root_path = os.path.dirname(
                    os.path.dirname(
                    os.path.dirname(
                    os.path.realpath(__file__))))
        log_path = os.path.join(root_path,"results",TIMESTAMP)
        fo.make_dir(log_path)
    else:
        figs = []

    fig = plt.figure(figsize=(5,3))
    axes = plt.gca()

    if x_metric is None:
        plt_title = y_metric
        plt.title(plt_title)
        data = navdata[y_metric]
        axes.scatter(range(data.shape[0]),data,s=5.)
        plt.xlabel("index")
        plt.ylabel(y_metric)
    else:
        plt_title = x_metric + " vs. " + y_metric
        plt.title(plt_title)
        axes.scatter(navdata[x_metric],navdata[y_metric],s=5.)
        plt.xlabel(x_metric)
        plt.ylabel(y_metric)

    axes.ticklabel_format(useOffset=False)


    if save: # pragma: no cover
        if prefix != "" and not prefix.endswith('_'):
            prefix += "_"
        plt_file = os.path.join(log_path,
                      prefix + plt_title.replace(" vs. ","_")  + ".png")

        fo.save_figure(fig, plt_file)

        # close previous figure
        plt.close(fig)

    else:
        figs.append(fig)

    if save: # pragma: no cover
        return None
    return figs


def plot_metric_by_constellation(navdata, metric, save=True, prefix=""):
    """Plot specific metric from a row of the NavData class.

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Instance of the NavData class
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

    if len(navdata.str_map[metric]):
        raise KeyError(metric + " is a non-numeric row, unable to plot.")
    if not isinstance(prefix, str):
        raise TypeError("Prefix must be a string.")
    if "signal_type" not in navdata.rows:
        raise KeyError("signal_type missing," \
                     + " try using" \
                     + " plot_metric() function call instead")
    if "sv_id" not in navdata.rows:
        raise KeyError("sv_id missing," \
                     + " try using" \
                     + " plot_metric() function call instead")

    if save: # pragma: no cover
        root_path = os.path.dirname(
                    os.path.dirname(
                    os.path.dirname(
                    os.path.realpath(__file__))))
        log_path = os.path.join(root_path,"results",TIMESTAMP)
        fo.make_dir(log_path)
    else:
        figs = []

    data = {}

    signal_types = navdata.get_strings("signal_type")
    sv_ids = navdata.get_strings("sv_id")

    time0 = navdata["gps_millis",0]/1000.

    for m_idx in range(navdata.shape[1]):
        if signal_types[m_idx] not in data:
            data[signal_types[m_idx]] = {}
        if sv_ids[m_idx] not in data[signal_types[m_idx]]:
            data[signal_types[m_idx]][sv_ids[m_idx]] = [[navdata["gps_millis",m_idx]/1000. - time0],
                                                  [navdata[metric,m_idx]]]
        else:
            data[signal_types[m_idx]][sv_ids[m_idx]][0].append(navdata["gps_millis",m_idx]/1000. - time0)
            data[signal_types[m_idx]][sv_ids[m_idx]][1].append(navdata[metric,m_idx])

    ####################################################################
    # BROKEN UP BY CONSTELLATION TYPE
    ####################################################################

    for signal_type, signal_data in data.items():
        fig = plt.figure(figsize=(5,3))
        axes = plt.gca()
        plt.title(get_signal_label(signal_type))

        for sv_name, sv_data in signal_data.items():
            axes.scatter(sv_data[0],sv_data[1],label=sv_name,s=5.)

        axes.ticklabel_format(useOffset=False)
        axes.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        plt.xlabel("time [s]")
        plt.ylabel(metric)
        plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

        if save: # pragma: no cover
            if prefix != "" and not prefix.endswith('_'):
                prefix += "_"
            plt_file = os.path.join(log_path, prefix + metric \
                     + "_" + signal_type + ".png")

            fo.save_figure(fig, plt_file)

            # close previous figure
            plt.close(fig)

        else:
            figs.append(fig)

    if save: # pragma: no cover
        return None
    return figs

def plot_skyplot(navdata, state_estimate, save=True, prefix=""):
    """Skyplot of data

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Instance of the NavData class
    state_estimate : gnss_lib_py.parsers.navdata.NavData
        Estimated receiver position in ECEF frame in meters and the
        estimated receiver clock bias also in meters as an instance of
        the NavData class with shape (4 x # unique timesteps) and
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
    if "signal_type" not in navdata.rows:
        raise KeyError("signal_type missing")
    if "sv_id" not in navdata.rows:
        raise KeyError("sv_id missing")
    if "x_sv_m" not in navdata.rows:
        raise KeyError("x_sv_m missing")
    if "y_sv_m" not in navdata.rows:
        raise KeyError("y_sv_m missing")
    if "z_sv_m" not in navdata.rows:
        raise KeyError("z_sv_m missing")
    if "x_rx_m" not in state_estimate.rows:
        raise KeyError("x_rx_m missing")
    if "y_rx_m" not in state_estimate.rows:
        raise KeyError("y_rx_m missing")
    if "z_rx_m" not in state_estimate.rows:
        raise KeyError("z_rx_m missing")

    local_coord = None

    skyplot_data = {}
    signal_types = list(navdata.get_strings("signal_type"))
    sv_ids = navdata.get_strings("sv_id")

    pos_sv_m = np.hstack((navdata["x_sv_m",:].reshape(-1,1),
                          navdata["y_sv_m",:].reshape(-1,1),
                          navdata["z_sv_m",:].reshape(-1,1)))

    for t_idx, timestep in enumerate(np.unique(navdata["gps_millis",:])):
        idxs = np.where(navdata["gps_millis",:] == timestep)[0]
        for m_idx in idxs:

            if signal_types[m_idx] not in skyplot_data:
                if "5" in signal_types[m_idx]:
                    continue
                skyplot_data[signal_types[m_idx]] = {}

            if local_coord is None:
                local_coord = LocalCoord.from_ecef(state_estimate[["x_rx_m","y_rx_m","z_rx_m"],t_idx])
            sv_ned = local_coord.ecef_to_ned(pos_sv_m[m_idx:m_idx+1,:])[0]

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
                axes.plot(sv_data[0][-1],sv_data[1][-1],c=color,
                        marker=marker, markersize=8,
                        label=get_signal_label(signal_type))
            else:
                axes.plot(sv_data[0][-1],sv_data[1][-1],c=color,
                        marker=marker, markersize=8)
            # axes.text(sv_data[0][-1], sv_data[1][-1], sv_name)

            s_idx += 1
        c_idx += 1

    axes.set_theta_zero_location('N')
    axes.set_theta_direction(-1)
    axes.set_yticks(range(0, 90+10, 30))    # Define the yticks
    axes.set_ylim(90,0)

    axes.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

    if save: # pragma: no cover
        root_path = os.path.dirname(
                    os.path.dirname(
                    os.path.dirname(
                    os.path.realpath(__file__))))
        log_path = os.path.join(root_path,"results",TIMESTAMP)
        fo.make_dir(log_path)
        if prefix != "" and not prefix.endswith('_'):
            prefix += "_"
        plt_file = os.path.join(log_path, prefix + "skyplot.png")

        fo.save_figure(fig, plt_file)

        # close previous figure
        plt.close(fig)

        return None

    return fig


def plot_residuals(navdata, save=True, prefix=""):
    """Plot residuals.

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Instance of the NavData class
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

    if "residuals" not in navdata.rows:
        raise KeyError("residuals missing, run solve_residuals().")
    if not isinstance(prefix, str):
        raise TypeError("Prefix must be a string.")
    if "signal_type" not in navdata.rows:
        raise KeyError("signal_type missing")
    if "sv_id" not in navdata.rows:
        raise KeyError("sv_id missing")
    if save: # pragma: no cover
        root_path = os.path.dirname(
                    os.path.dirname(
                    os.path.dirname(
                    os.path.realpath(__file__))))
        log_path = os.path.join(root_path,"results",TIMESTAMP)
        fo.make_dir(log_path)
    else:
        figs = []

    residual_data = {}
    signal_types = navdata.get_strings("signal_type")
    sv_ids = navdata.get_strings("sv_id")

    time0 = navdata["gps_millis",0]/1000.

    for m_idx in range(navdata.shape[1]):
        if signal_types[m_idx] not in residual_data:
            residual_data[signal_types[m_idx]] = {}
        if sv_ids[m_idx] not in residual_data[signal_types[m_idx]]:
            residual_data[signal_types[m_idx]][sv_ids[m_idx]] = [[navdata["gps_millis",m_idx]/1000. - time0],
                        [navdata["residuals",m_idx]]]
        else:
            residual_data[signal_types[m_idx]][sv_ids[m_idx]][0].append(navdata["gps_millis",m_idx]/1000. - time0)
            residual_data[signal_types[m_idx]][sv_ids[m_idx]][1].append(navdata["residuals",m_idx])

    ####################################################################
    # BROKEN UP BY CONSTELLATION TYPE
    ####################################################################


    for signal_type, signal_residuals in residual_data.items():
        fig = plt.figure(figsize=(5,3))

        plt.title(get_signal_label(signal_type))
        signal_type_svs = list(signal_residuals.keys())

        for sv_name, sv_data in signal_residuals.items():
            plt.plot(sv_data[0], sv_data[1],
                     label = get_signal_label(signal_type) + " " + str(sv_name))
        axes = plt.gca()
        axes.ticklabel_format(useOffset=False)
        axes.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        plt.ylim(-100.,100.)
        plt.xlabel("time [s]")
        plt.ylabel("residiual [m]")
        plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

        if save: # pragma: no cover
            if prefix != "" and not prefix.endswith('_'):
                prefix += "_"
            plt_file = os.path.join(log_path, prefix + "residuals_" \
                     + signal_type + ".png")

            fo.save_figure(fig, plt_file)

            # close previous figure
            plt.close(fig)
        else:
            figs.append(fig)

    if save: # pragma: no cover
        return None
    return figs

def get_signal_label(signal_name_raw):
    """Return signal name with better formatting for legend.

    Parameters
    ----------
    signal_name_raw : string
        Signal name with underscores between parts of singal type.
        For example, GPS_L1

    Returns
    -------
    signal_label : string
        Properly formatted signal label

    """

    signal_label = signal_name_raw.replace("_"," ")

    # replace with lowercase "i" for Beidou "I" signals for more legible
    # name in the legend
    if signal_label[-1] == "I":
        signal_label = signal_label[:-1] + "i"

    return signal_label
