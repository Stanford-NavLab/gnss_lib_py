"""Visualization functions of GNSS data.

"""

__authors__ = "D. Knowles"
__date__ = "27 Jan 2022"

import os
import pathlib

import numpy as np
from cycler import cycler
import plotly.express as px
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb, ListedColormap

import gnss_lib_py.utils.file_operations as fo
from gnss_lib_py.utils.coordinates import ecef_to_el_az

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

GNSS_ORDER = ["gps","glonass","galileo","beidou","qzss","irnss","sbas",
              "unkown"]

mpl.rcParams['axes.prop_cycle'] = (cycler(color=STANFORD_COLORS) \
                                +  cycler(marker=MARKERS))
TIMESTAMP = fo.get_timestamp()

def plot_metric(navdata, *args, groupby=None, title=None, save=True,
                prefix="", fname=None, **kwargs):
    """Plot specific metric from a row of the NavData class.

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Instance of the NavData class
    *args : tuple
        Tuple of row names that are to be plotted. If one is given, that
        value is plotted on the y-axis. If two values are given, the
        first is plotted on the x-axis and the second on the y-axis.
    grouby : string
        Row name by which to groub and label plots
    save : bool
        Saves figure if true to file specified by fname or defaults
        to the Results folder otherwise.
    prefix : string
        File prefix to add to filename.
    fname : string or path-like
        Path to save figure to. If not None, fname is passed directly
        to matplotlib's savefig fname parameter and prefix will be
        overwritten.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
         Figure of plotted metrics.

    """

    x_metric, y_metric = _parse_metric_args(navdata, *args)

    if groupby is not None:
        navdata.in_rows(groupby)
    if not isinstance(prefix, str):
        raise TypeError("Prefix must be a string.")

    fig, axes = _get_new_fig()

    if x_metric is None:
        if title is None:
            title = _get_label({y_metric:y_metric})
        plt.xlabel("Index")
        if groupby is not None:
            for group in np.unique(navdata[groupby]):
                subset = navdata.where(groupby,group)
                y_data = np.atleast_1d(subset[y_metric])
                axes.plot(range(len(y_data)), y_data,
                          label=_get_label({groupby:group}), **kwargs)
        else:
            y_data = np.atleast_1d(navdata[y_metric])
            axes.plot(range(len(y_data)), y_data,
                         **kwargs)
    else:
        if title is None:
            title = _get_label({x_metric:x_metric}) + " vs. " \
                  + _get_label({y_metric:y_metric})
        plt.xlabel(_get_label({x_metric:x_metric}))
        if groupby is not None:
            all_groups = np.unique(navdata[groupby])
            if groupby == "gnss_id":
                all_groups = _sort_gnss_ids(all_groups)
            for group in all_groups:
                subset = navdata.where(groupby,group)
                x_data = np.atleast_1d(subset[x_metric])
                y_data = np.atleast_1d(subset[y_metric])
                axes.plot(x_data, y_data,
                          label=_get_label({groupby:group}),**kwargs)
        else:
            x_data = np.atleast_1d(navdata[x_metric])
            y_data = np.atleast_1d(navdata[y_metric])
            axes.plot(x_data, y_data,
                      **kwargs)

    handles, _ = axes.get_legend_handles_labels()
    if len(handles) > 0:
        plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1),
                   title=_get_label({groupby:groupby}))

    plt.title(title)
    plt.ylabel(_get_label({y_metric:y_metric}))
    fig.tight_layout()

    if save: # pragma: no cover
        _save_figure(fig, title, prefix, fname)
    return fig

def plot_metric_by_constellation(navdata, *args, save=True, prefix="",
                                 fname=None, **kwargs):
    """Plot specific metric from a row of the NavData class.

    Breaks up metrics by constellation names in "gnss_id" and
    additionally "signal_type" if the "signal_type" row exists.

    Plots will include a legend with satellite ID if the "sv_id" row
    is present in navdata.

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Instance of the NavData class. Must include ``gnss_id`` row and
        optionally ``signal_type`` and ``sv_id`` for increased
        labelling.
    *args : tuple
        Tuple of row names that are to be plotted. If one is given, that
        value is plotted on the y-axis. If two values are given, the
        first is plotted on the x-axis and the second on the y-axis.
    save : bool
        Saves figure if true to file specified by ``fname`` or defaults
        to the Results folder otherwise.
    prefix : string
        File prefix to add to filename.
    fname : string or path-like
        Path to save figure to. If not None, ``fname`` is passed
        directly to matplotlib's savefig fname parameter and prefix will
        be overwritten.

    Returns
    -------
    fig : list of matplotlib.pyplot.Figure objects
         List of figures of plotted metrics.

    """

    x_metric, y_metric = _parse_metric_args(navdata, *args)

    if not isinstance(prefix, str):
        raise TypeError("Prefix must be a string.")
    if "gnss_id" not in navdata.rows:
        raise KeyError("gnss_id row missing," \
                     + " try using" \
                     + " the plot_metric() function call instead")

    figs = []
    for constellation in _sort_gnss_ids(np.unique(navdata["gnss_id"])):
        const_subset = navdata.where("gnss_id",constellation)

        if prefix == "":
            prefix = constellation + "_"
        else:
            if prefix != "" and not prefix.endswith('_'):
                prefix += "_"
            prefix += constellation + "_"

        if "signal_type" in const_subset.rows:
            for signal in np.unique(const_subset["signal_type"]):
                prefix += signal + "_"
                title = _get_label({"gnss_id":constellation,"signal_type":signal})
                signal_subset = navdata.where("signal_type",signal)
                if "sv_id" in signal_subset.rows:
                    # group by sv_id
                    fig = plot_metric(signal_subset,x_metric,y_metric,
                                      groupby="sv_id", title=title,
                                      save=save, prefix=prefix,
                                      fname=fname, **kwargs)
                    figs.append(fig)
                else:
                    fig = plot_metric(signal_subset,x_metric,y_metric,
                                      title=title, save=save,
                                      prefix=prefix, fname=fname,
                                      **kwargs)
                    figs.append(fig)
        else:
            title = _get_label({"gnss_id":constellation})
            if "sv_id" in const_subset.rows:
                # group by sv_id
                fig = plot_metric(const_subset,x_metric,y_metric,
                                  groupby="sv_id", title=title,
                                  save=save, prefix=prefix, fname=fname,
                                  **kwargs)
                figs.append(fig)
            else:
                fig = plot_metric(const_subset,x_metric,y_metric,
                                  title=title, save=save, prefix=prefix,
                                  fname=fname, **kwargs)
                figs.append(fig)

    return figs

def plot_skyplot(navdata, receiver_state, save=True, prefix="",
                 fname=None, **kwargs):
    """Skyplot of satellite positions relative to receiver.

    First adds ``el_sv_deg`` and ``az_sv_deg`` rows to navdata if they
    do not yet exist.

    Breaks up satellites by constellation names in ``gnss_id`` and the
    ``sv_id`` if the row is present in navdata.

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Instance of the NavData class. Must include ``gps_millis`` as
        well as satellite ECEF positions as ``x_sv_m``, ``y_sv_m``,
        ``z_sv_m``, ``gnss_id`` and ``sv_id``.
    receiver_state : gnss_lib_py.parsers.navdata.NavData
        Either estimated or ground truth receiver position in ECEF frame
        in meters as an instance of the NavData class with the
        following rows: ``x_*_m``, ``y_*_m``, ``z_*_m``, ``gps_millis``.
    save : bool
        Saves figure if true to file specified by ``fname`` or defaults
        to the Results folder otherwise.
    prefix : string
        File prefix to add to filename.
    fname : string or path-like
        Path to save figure to. If not None, ``fname`` is passed
        directly to matplotlib's savefig fname parameter and prefix will
        be overwritten.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Figure object of skyplot, returns None if save set to True.

    """

    if not isinstance(prefix, str):
        raise TypeError("Prefix must be a string.")
    # check for missing rows
    navdata.in_rows(["gps_millis","x_sv_m","y_sv_m","z_sv_m",
                     "gnss_id","sv_id"])
    receiver_state.in_rows(["gps_millis"])

    # check for receiver_state indexes
    rx_idxs = {"x_*_m" : [],
               "y_*_m" : [],
               "z_*_m" : [],
               }
    for name, indexes in rx_idxs.items():
        indexes = [row for row in receiver_state.rows
                      if row.startswith(name.split("*",maxsplit=1)[0])
                       and row.endswith(name.split("*",maxsplit=1)[1])]
        if len(indexes) > 1:
            raise KeyError("Multiple possible row indexes for " \
                         + name \
                         + ". Unable to resolve for plot_skyplot().")
        if len(indexes) == 0:
            raise KeyError("Missing required " + name + " row for " \
                        + "plot_skyplot().")
        # must call dictionary to avoid pass by value
        rx_idxs[name] = indexes[0]

    if "el_sv_deg" not in navdata.rows or "az_sv_deg" not in navdata.rows:
        sv_el_az = None

        for timestamp, _, navdata_subset in navdata.loop_time("gps_millis"):

            pos_sv_m = navdata_subset[["x_sv_m","y_sv_m","z_sv_m"]].T

            # find time index for receiver_state NavData instance
            rx_t_idx = np.argmin(np.abs(receiver_state["gps_millis"] - timestamp))

            pos_rx_m = receiver_state[[rx_idxs["x_*_m"],
                                       rx_idxs["y_*_m"],
                                       rx_idxs["z_*_m"]],
                                       rx_t_idx].reshape(1,-1)

            timestep_el_az = ecef_to_el_az(pos_rx_m, pos_sv_m)

            if sv_el_az is None:
                sv_el_az = timestep_el_az.T
            else:
                sv_el_az = np.hstack((sv_el_az,timestep_el_az.T))

        navdata["el_sv_deg"] = sv_el_az[0,:]
        navdata["az_sv_deg"] = sv_el_az[1,:]

    # create new figure
    fig = plt.figure(figsize=(8,6))
    axes = fig.add_subplot(111, projection='polar')

    navdata = navdata.copy()
    navdata["az_sv_rad"] = np.radians(navdata["az_sv_deg"])

    for c_idx, constellation in enumerate(_sort_gnss_ids(np.unique(navdata["gnss_id"]))):
        const_subset = navdata.where("gnss_id",constellation)
        color = "C" + str(c_idx % len(STANFORD_COLORS))
        cmap = _new_cmap(to_rgb(color))
        marker = MARKERS[c_idx % len(MARKERS)]

        # iterate through each satellite
        for sv_idx, sv_name in enumerate(np.unique(const_subset["sv_id"])):
            sv_subset = const_subset.where("sv_id",sv_name)
            # remove np.nan values caused by potentially faulty data
            sv_subset = sv_subset.where("az_sv_rad",np.nan,"neq")
            sv_subset = sv_subset.where("el_sv_deg",np.nan,"neq")
            if len(sv_subset) == 0:
                continue
            # only plot ~ 50 points for each sat to decrease time
            # it takes to plot these line collections
            step = max(1,int(len(sv_subset)/50.))
            points = np.array([np.atleast_1d(sv_subset["az_sv_rad"])[::step],
                               np.atleast_1d(sv_subset["el_sv_deg"])[::step]]).T
            points = np.reshape(points,(-1, 1, 2))
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(0,len(segments))
            local_coord = LineCollection(segments, cmap=cmap,
                            norm=norm, linewidths=(4,),
                            array = range(len(segments)))
            axes.add_collection(local_coord)
            if sv_idx == 0:
                # plot with label
                axes.plot(np.atleast_1d(sv_subset["az_sv_rad"])[-1],
                          np.atleast_1d(sv_subset["el_sv_deg"])[-1],
                          c=color, marker=marker, markersize=8,
                    label=_get_label({"gnss_id":constellation}))
            else:
                # plot without label
                axes.plot(np.atleast_1d(sv_subset["az_sv_rad"])[-1],
                          np.atleast_1d(sv_subset["el_sv_deg"])[-1],
                          c=color, marker=marker, markersize=8)
            axes.text(np.atleast_1d(sv_subset["az_sv_rad"])[-1]\
                    + 3.*np.radians(np.cos(np.atleast_1d(sv_subset["az_sv_rad"])[-1])),
                      np.atleast_1d(sv_subset["el_sv_deg"])[-1]\
                    - 3.*np.sin(np.atleast_1d(sv_subset["az_sv_rad"])[-1]),
                      str(int(sv_name)),
                      backgroundcolor=(1.,1.,1.,0.2))

    # updated axes for skyplot graph specifics
    axes.set_theta_zero_location('N')
    axes.set_theta_direction(-1)
    axes.set_yticks(range(0, 60+10, 30))    # Define the yticks
    axes.set_yticklabels(['',r'$30\degree$',r'$60\degree$'])
    axes.set_ylim(90,0)

    handles, _ = axes.get_legend_handles_labels()
    if len(handles) > 0:
        axes.legend(loc="upper left", bbox_to_anchor=(1.05, 1),
                   title=_get_label({"sv_id":"sv_id"}))

    fig.tight_layout()

    if save: # pragma: no cover
        _save_figure(fig, "skyplot", prefix=prefix, fnames=fname)
    return fig

def map_lla(*args, save=True, prefix="", **kwargs):
    """Map trajectories.

    Parameters
    ----------
    *args : tuple
        Tuple of gnss_lib_py.parsers.navdata.NavData objects. The
        NavData objects should include.
    save : bool
        Save figure if true, otherwise returns figure object. Defaults
        to saving the figure in the Results folder.
    prefix : string
        File prefix to add to filename.
    mapbox_style : str
        Can optionally be included as one of the ``**kwargs``
        Free options include ``open-street-map``, ``white-bg``,
        ``carto-positron``, ``carto-darkmatter``, ``stamen-terrain``,
        ``stamen-toner``, and ``stamen-watercolor``.

    Returns
    -------
    figs : list
        List of matplotlib.pyplot.figure objects of residuels, returns
        None if save set to True.

    """
    # TODO: add description about what NavData objects should include

    if save: # pragma: no cover
        root_path = os.path.dirname(
                    os.path.dirname(
                    os.path.dirname(
                    os.path.realpath(__file__))))
        log_path = os.path.join(root_path,"results",TIMESTAMP)
        fo.make_dir(log_path)

    fig = None

    for traj_data in args:
        lat_row_name = [s for s in traj_data.rows if "lat" in s]
        lon_row_name = [s for s in traj_data.rows if "lon" in s]
        # TODO: raise warning if non existent lat or if more than one.

        if fig is None:
            fig = px.scatter_mapbox(traj_data,
                                    lat=traj_data[lat_row_name],
                                    lon=traj_data[lon_row_name],
                                    size=1*np.ones(traj_data[lat_row_name].size, dtype=np.int),

                                    )
        else:
            fig2 = px.scatter_mapbox(traj_data,
                                    lat=traj_data[lat_row_name],
                                    lon=traj_data[lon_row_name],
                                    color=traj_data['gps_millis'],
                                    size=5*np.ones(traj_data[lat_row_name].size, dtype=np.int),
                                    # color= "b",
                                    )
            fig.add_trace(fig2.data[0])

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(**kwargs)
#     fig.update_layout(title_text="Ground Truth Tracks of Android Smartphone GNSS Dataset")
#     fig.legend()


    if save: # pragma: no cover
        if prefix != "" and not prefix.endswith('_'):
            prefix += "_"
        plt_file = os.path.join(log_path, prefix + "map.png")

        fig.write_image(plt_file)

        return None

    return fig

def close_figures(figs):
    """Closes figures.

    Parameters
    ----------
    figs : list or matplotlib.pyplot.figure
        List of figures or single matplotlib figure object.

    """

    if isinstance(figs,plt.Figure):
        plt.close(figs)
    elif isinstance(figs, list):
        for fig in figs:
            plt.close(fig)
    else:
        raise TypeError("Must be either a single figure or list of figures.")

def _get_new_fig():
    """

    fig : matplotlib.pyplot.figure
        Default NavData figure.
    axes : matplotlib.pyplot.axes
        Default NavData axes.

    """

    fig = plt.figure()
    axes = plt.gca()

    axes.ticklabel_format(useOffset=False)
    fig.autofmt_xdate() # rotate x labels automatically

    return fig, axes

def _get_label(inputs):
    """Return label/title name from input dictionary.

    Parameters
    ----------

    inputs : dict
        Dictionary of {row_name : row_value} pairs to create name from.

    Returns
    -------
    label : string
        Properly formatted label/title for use in graphs.

    """

    if not isinstance(inputs,dict):
        raise TypeError("_get_label input must be dictionary.")

    label = ""
    for key, value in inputs.items():

        if len(label) != 0: # add space between multiple inputs
            value = " " + value

        if not isinstance(value,str): # convert numbers/arrays to string
            value = str(value)

        try: # convert to integer if a numeric value
            value = str(int(float(value)))
        except ValueError:
            pass

        value = value.upper()
        value = value.replace("_"," ")

        if key == "gnss_id": # use GNSS specific capitalization
            constellation_map = {"GALILEO" : "Galileo",
                                 "BEIDOU" : "BeiDou"
                                 }
            for old_value, new_value in constellation_map.items():
                value = value.replace(old_value,new_value)

        if key == "signal_type":
            # replace with lowercase "i" for Beidou "I" signals for more
            # legible name in the legend
            if value[-1] == "I":
                value = value[:-1] + "i"

        label += value

    return label

def _sort_gnss_ids(unsorted_gnss_ids):
    """Sort constellations by chronological availability.

    Order defined by `GNSS_ORDER` variable in header.

    Parameters
    ----------
    unsorted_gnss_ids : list or array-like of strings.
        Unsorted constellation names.

    Returns
    -------
    sorted_gnss_ids : list or array-like of strings.
        Sorted constellation names.

    """

    sorted_gnss_ids = []
    unsorted_gnss_ids = list(unsorted_gnss_ids)
    for gnss in GNSS_ORDER:
        if gnss in unsorted_gnss_ids:
            unsorted_gnss_ids.remove(gnss)
            sorted_gnss_ids.append(gnss)
    sorted_gnss_ids += sorted(unsorted_gnss_ids)

    return sorted_gnss_ids

def _save_figure(figures, titles=None, prefix="", fnames=None): # pragma: no cover
    """Saves figures to file.

    Parameters
    ----------
    figures : single or list of List of matplotlib.pyplot.figure objects
        Figures to be saved and closed.
    titles : string, path-like or list of strings
        Titles for all plots.
    prefix : string
        File prefix to add to filename.
    fnames : single or list of string or path-like
        Path to save figure to. If not None, fname is passed directly
        to matplotlib's savefig fname parameter and prefix will be
        overwritten.

    """

    if isinstance(figures, plt.Figure):
        figures = [figures]
    if isinstance(titles,str) or titles is None:
        titles = [titles]
    if type(fnames) in (str, pathlib.Path) or fnames is None:
        fnames = [fnames]

    for fig_idx, figure in enumerate(figures):

        if fnames[fig_idx] is None:
            # create results folder if it does not yet exist.
            root_path = os.path.dirname(
                        os.path.dirname(
                        os.path.dirname(
                        os.path.realpath(__file__))))
            log_path = os.path.join(root_path,"results",TIMESTAMP)
            fo.make_dir(log_path)

            # make name path friendly
            title = titles[fig_idx]
            title.replace(" ","_")
            title.replace(".","")
            title.replace("vs","")

            if prefix != "" and not prefix.endswith('_'):
                prefix += "_"
            fname = os.path.join(log_path, prefix + title \
                                                  + ".png")
        else:
            fname = fnames[fig_idx]

        figure.savefig(fname,
                       dpi=300.,
                       format="png",
                       bbox_inches="tight")

def _parse_metric_args(navdata, *args):
    """Parses arguments and raises error if metrics are nonnumeric.

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Instance of the NavData class
    *args : tuple
        Tuple of row names that are to be plotted. If one is given, that
        value is plotted on the y-axis. If two values are given, the
        first is plotted on the x-axis and the second on the y-axis.

    Returns
    -------
    x_metric : string
        Metric to be plotted on y-axis if y_metric is None, otherwise
        x_metric is plotted on x axis.
    y_metric : string or None
        y_metric is plotted on the y axis.

    """

    # parse arguments
    if len(args)==1:
        x_metric = None
        y_metric = args[0]
    elif len(args)==2:
        x_metric = args[0]
        y_metric = args[1]
    else:
        raise ValueError("Cannot plot more than one pair of x-y values")
    for metric in [x_metric, y_metric]:
        if metric is not None and navdata.is_str(metric):
            raise KeyError(metric + " is a non-numeric row." \
                         + "Unable to plot with plot_metric().")

    return x_metric, y_metric

def _new_cmap(rgb_color):
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
