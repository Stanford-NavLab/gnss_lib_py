"""Visualization functions for GNSS data.

"""

__authors__ = "D. Knowles"
__date__ = "27 Jan 2022"

import os
import pathlib
from math import floor
from multiprocessing import Process

import numpy as np
import pandas as pd
from cycler import cycler
import plotly.express as px
import plotly.graph_objects as go
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb, ListedColormap

import gnss_lib_py.utils.file_operations as fo
from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.utils.coordinates import add_el_az

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
              "unknown"]

mpl.rcParams['axes.prop_cycle'] = (cycler(color=STANFORD_COLORS) \
                                +  cycler(marker=MARKERS))

def plot_metric(navdata, *args, groupby=None, avg_y=False, fig=None,
                title=None, save=False, prefix="", fname=None,
                markeredgecolor="k", markeredgewidth=0.2, **kwargs):
    """Plot specific metric from a row of the NavData class.

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Instance of the NavData class
    *args : tuple
        Tuple of row names that are to be plotted. If one is given, that
        value is plotted on the y-axis. If two values are given, the
        first is plotted on the x-axis and the second on the y-axis.
    groupby : string
        Row name by which to group and label plots.
    avg_y : bool
        Whether or not to average across the y values for each x
        timestep when doing groupby
    fig : matplotlib.pyplot.Figure
         Previous figure on which to add current plotting. Default of
         None plots on a new figure.
    title : string
        Title for the plot.
    save : bool
        Saves figure if true to file specified by fname or defaults
        to the Results folder otherwise.
    prefix : string
        File prefix to add to filename.
    fname : string or path-like
        Path to save figure. If not None, fname is passed directly
        to matplotlib's savefig fname parameter and prefix will be
        overwritten.
    markeredgecolor : color
        Marker edge color.
    markeredgewidth : float
        Marker edge width.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
         Figure of plotted metrics.

    """

    if not isinstance(navdata,NavData):
        raise TypeError("first arg to plot_metrics must be a "\
                          + "NavData object.")

    x_metric, y_metric = _parse_metric_args(navdata, *args)

    if groupby is not None:
        navdata.in_rows(groupby)
    if not isinstance(prefix, str):
        raise TypeError("Prefix must be a string.")

    # create a new figure if none provided
    fig, axes = _get_new_fig(fig)

    if x_metric is None:
        x_data = None
        xlabel = "INDEX"
        if title is None:
            title = _get_label({y_metric:y_metric})
    else:
        if title is None:
            title = _get_label({y_metric:y_metric}) + " vs. " \
                  + _get_label({x_metric:x_metric})
        xlabel = _get_label({x_metric:x_metric})

    if groupby is not None:
        all_groups = np.unique(navdata[groupby])
        if groupby == "gnss_id":
            all_groups = _sort_gnss_ids(all_groups)
        for group in all_groups:
            subset = navdata.where(groupby,group)
            y_data = np.atleast_1d(subset[y_metric])
            if x_metric is None:
                x_data = range(len(y_data))
            else:
                x_data = np.atleast_1d(subset[x_metric])
            if avg_y:
                # average y values for each x
                x_unique = sorted(np.unique(x_data))
                y_avg = []
                for x_val in x_unique:
                    x_idxs = np.argwhere(x_data==x_val)
                    y_avg.append(np.mean(y_data[x_idxs]))
                x_data = x_unique
                y_data = y_avg
                # change name
                group = str(group) + "_avg"
            axes.plot(x_data, y_data,
                      label=_get_label({groupby:group}),
                      markeredgecolor = markeredgecolor,
                      markeredgewidth = markeredgewidth,
                      **kwargs)
    else:
        y_data = np.atleast_1d(navdata[y_metric])
        if x_metric is None:
            x_data = range(len(y_data))
        else:
            x_data = np.atleast_1d(navdata[x_metric])
        axes.plot(x_data, y_data,
                  markeredgecolor = markeredgecolor,
                  markeredgewidth = markeredgewidth,
                  **kwargs)

    handles, _ = axes.get_legend_handles_labels()
    if len(handles) > 0:
        plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1),
                   title=_get_label({groupby:groupby}))

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(_get_label({y_metric:y_metric}))
    fig.set_layout_engine(layout="tight")

    if save: # pragma: no cover
        _save_figure(fig, title, prefix, fname)
    return fig

def plot_metric_by_constellation(navdata, *args, save=False, prefix="",
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

    if not isinstance(navdata,NavData):
        raise TypeError("first arg to plot_metric_by_constellation "\
                          + "must be a NavData object.")

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

        if "signal_type" in const_subset.rows:
            for signal in np.unique(const_subset["signal_type"]):
                title = _get_label({"gnss_id":constellation,"signal_type":signal})
                signal_subset = const_subset.where("signal_type",signal)
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

def plot_skyplot(navdata, receiver_state,
                 save=False, prefix="", fname=None,
                 add_sv_id_label=True, step = "auto", trim_options=None):
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
        following rows: ``x_rx*_m``, ``y_rx*_m``, ``z_rx*_m``, ``gps_millis``.
    save : bool
        Saves figure if true to file specified by ``fname`` or defaults
        to the Results folder otherwise.
    prefix : string
        File prefix to add to filename.
    fname : string or path-like
        Path to save figure to. If not None, ``fname`` is passed
        directly to matplotlib's savefig fname parameter and prefix will
        be overwritten.
    add_sv_id_label : bool
        If the ``sv_id`` row is available, will add SV ID label near the
        satellite trail.
    step : int or "auto"
        Skyplot plotting is sped up by only plotting a portion of the
        satellite trajectories. If default is set to "auto" then it will
        plot a maximum of 50 points across each satellite trajectory. If
        the step variable is set to a positive integer ``n`` then only
        every nth point will be used in the trajectory. Setting the
        steps variable to 1 will plot every satellite trajectory point
        and may be slow to plot.
    trim_options : None or dict
        The ``trim_options`` variables gives control for line segments
        being trimmed between satellite points. For example, if 24 hours
        of a satellite is plotted, often the satellite will come in and
        out of view and the segment between when it was lost from view
        and when the satellite comes back in view should be trimmed.
        If trim_options is set to the default of None, then the default
        is set of trimming according to az_and_el and gps_millis. The
        current options for the trim_options dictionary are listed here.
        {"az" : az_limit} means that if at two timesteps the azimuth
        difference in degrees is greater than az_limit, then the line
        segment will be trimmed.
        {"az_and_el" : (az_limit,el_limit)} means that if at two
        timesteps the azimuth difference in degrees is greater than
        az_limit as well as the average of the elevation angle across
        the two timesteps is less than el_limit in degrees, then the
        line segment will be trimmed. The el_limit is because satellites
        near 90 degrees elevation can traverse large amounts of degrees
        in azimuth in a valid trajectory but at low elevations should
        not have such large azimuth changes quickly.
        {"gps_millis",gps_millis_limit} means that line segments will be
        trimmed if the milliseconds between the two points is larger
        than the gps_millis_limit. This option only works if the
        gps_millis row is included in the ``navdata`` variable input.
        Default options for the trim options are :code:`"az_and_el" : (15.,30.)`
        and :code:`"gps_millis" : 3.6E6`.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Figure object of skyplot.

    """

    if not isinstance(navdata,NavData):
        raise TypeError("first arg to plot_skyplot "\
                          + "must be a NavData object.")

    if not isinstance(prefix, str):
        raise TypeError("Prefix must be a string.")

    # add elevation and azimuth data.
    add_el_az(navdata, receiver_state, inplace=True)

    # create new figure
    fig = plt.figure(figsize=(6,4.5))
    axes = fig.add_subplot(111, projection='polar')

    navdata = navdata.copy()
    navdata["az_sv_rad"] = np.radians(navdata["az_sv_deg"])
    # remove SVs below horizon
    navdata = navdata.where("el_sv_deg",0,"geq")
    # remove np.nan values caused by potentially faulty data
    navdata = navdata.where("az_sv_rad",np.nan,"neq")
    navdata = navdata.where("el_sv_deg",np.nan,"neq")

    for c_idx, constellation in enumerate(_sort_gnss_ids(np.unique(navdata["gnss_id"]))):
        const_subset = navdata.where("gnss_id",constellation)
        color = "C" + str(c_idx % len(STANFORD_COLORS))
        cmap = _new_cmap(to_rgb(color))
        marker = MARKERS[c_idx % len(MARKERS)]
        const_label_created = False

        # iterate through each satellite
        for sv_name in np.unique(const_subset["sv_id"]):
            sv_subset = const_subset.where("sv_id",sv_name)

            # only plot ~ 50 points for each sat to decrease time
            # it takes to plot these line collections if step == "auto"
            if isinstance(step,str) and step == "auto":
                step = max(1,int(len(sv_subset)/50.))
            elif isinstance(step, int):
                step = max(1,step)
            else:
                raise TypeError("step varaible must be 'auto' or int")
            points = np.array([np.atleast_1d(sv_subset["az_sv_rad"])[::step],
                               np.atleast_1d(sv_subset["el_sv_deg"])[::step]]).T
            points = np.reshape(points,(-1, 1, 2))
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(0,len(segments))

            if trim_options is None:
                trim_options = {
                                "az_and_el" : (15.,30.),
                                "gps_millis" : 3.6E6,
                                }
            plotted_idxs = np.array([True] * len(segments))

            if "az" in trim_options and len(segments) > 2:
                # ignore segments that cross more than az_limit degrees
                # in azimuth between timesteps
                az_limit = np.radians(trim_options["az"])
                az_idxs = ~((np.abs(np.diff(np.unwrap(segments[:,:,0]))) >= az_limit)[:,0])
                plotted_idxs = np.bitwise_and(plotted_idxs, az_idxs)
            if "az_and_el" in trim_options and len(segments) > 2:
                # ignore segments that cross more than az_limit degrees
                # in azimuth between timesteps and are at an elevation
                # less than el_limit degrees.
                # These satellites are assumed to be the satellites
                # coming in and out of view in a later part of the orbit
                az_limit = np.radians(trim_options["az_and_el"][0])
                el_limit = trim_options["az_and_el"][1]
                az_and_el_idxs = ~(((np.abs(np.diff(np.unwrap(segments[:,:,0]))) >= az_limit)[:,0]) \
                                 & (np.mean(segments[:,:,1],axis=1) <= el_limit))
                plotted_idxs = np.bitwise_and(plotted_idxs, az_and_el_idxs)
            if "gps_millis" in trim_options and "gps_millis" in sv_subset.rows \
                and len(segments) > 2:
                # ignore segments if there is more than gps_millis_limit
                # milliseconds between the time segments
                gps_millis_limit = trim_options["gps_millis"]

                all_times = np.atleast_2d(sv_subset["gps_millis"][::step]).T
                point_times = np.concatenate([all_times[:-1],all_times[1:]],
                                              axis=1)
                gps_millis_idxs = (np.abs(np.diff(point_times)) <= gps_millis_limit)[:,0]
                plotted_idxs = np.bitwise_and(plotted_idxs, gps_millis_idxs)

            segments = segments[list(plotted_idxs)]

            local_coord = LineCollection(segments, cmap=cmap,
                            norm=norm, linewidths=(4,),
                            array = range(len(segments)))
            axes.add_collection(local_coord)
            if not const_label_created:
                # plot with label
                axes.plot(np.atleast_1d(sv_subset["az_sv_rad"])[-1],
                          np.atleast_1d(sv_subset["el_sv_deg"])[-1],
                          c=color, marker=marker, markersize=8,
                    label=_get_label({"gnss_id":constellation}))
                const_label_created = True
            else:
                # plot without label
                axes.plot(np.atleast_1d(sv_subset["az_sv_rad"])[-1],
                          np.atleast_1d(sv_subset["el_sv_deg"])[-1],
                          c=color, marker=marker, markersize=8)
            if add_sv_id_label:
                # offsets move label to the right of marker
                az_offset = 3.*np.radians(np.cos(np.atleast_1d(sv_subset["az_sv_rad"])[-1]))
                el_offset = -3.*np.sin(np.atleast_1d(sv_subset["az_sv_rad"])[-1])
                axes.text(np.atleast_1d(sv_subset["az_sv_rad"])[-1] \
                          + az_offset,
                          np.atleast_1d(sv_subset["el_sv_deg"])[-1] \
                          + el_offset,
                          str(int(sv_name)),
                          )

    # updated axes for skyplot graph specifics
    axes.set_theta_zero_location('N')
    axes.set_theta_direction(-1)
    axes.set_yticks(range(0, 60+10, 30))    # Define the yticks
    axes.set_yticklabels(['',r'$30\degree$',r'$60\degree$'])
    axes.set_ylim(90,0)

    handles, _ = axes.get_legend_handles_labels()
    if len(handles) > 0:
        axes.legend(loc="upper left", bbox_to_anchor=(1.05, 1),
                   title=_get_label({"constellation":"constellation"}))

    fig.set_layout_engine(layout='tight')

    if save: # pragma: no cover
        _save_figure(fig, "skyplot", prefix=prefix, fnames=fname)
    return fig

def plot_map(*args, sections=0, save=False, prefix="",
             fname=None, width=730, height=520, **kwargs):
    """Map lat/lon trajectories on map.

    By increasing the ``sections`` parameter, it is possible to output
    multiple zoom sections of the trajectories to see finer details.

    Parameters
    ----------
    *args : gnss_lib_py.parsers.navdata.NavData
        Tuple of gnss_lib_py.parsers.navdata.NavData objects. The
        NavData objects should include row names for both latitude and
        longitude in the form of ```lat_*_deg`` and ``lon_*_deg``.
        Must also include ``gps_millis`` if sections >= 2.
    sections : int
        Number of zoomed in sections to make of data. Will only output
        additional plots if sections >= 2. Creates sections by equal
        timestamps using the ``gps_millis`` row.
    save : bool
        Save figure if true. Defaults to saving the figure in the
        Results folder.
    prefix : string
        File prefix to add to filename.
    fname : string or path-like
        Path to save figure to. If not None, ``fname`` is passed
        directly to matplotlib's savefig fname parameter and prefix will
        be overwritten.
    width : int
        Figure width in pixels.
    height : int
        Figure height in pixels.
    mapbox_style : str
        Can optionally be included as one of the ``**kwargs``
        Free options include ``open-street-map``, ``white-bg``,
        ``carto-positron``, ``carto-darkmatter``, ``stamen-terrain``,
        ``stamen-toner``, and ``stamen-watercolor``.

    Returns
    -------
    figs : single or list of plotly.graph_objects.Figure
        Returns single plotly.graph_objects.Figure object if sections is
        <= 1, otherwise returns list of Figure objects containing full
        trajectory as well as zoomed in sections.

    """

    figure_df = None        # plotly works best passing in DataFrame
    color_discrete_map = {} # discrete color map

    for idx, traj_data in enumerate(args):
        if not isinstance(traj_data, NavData):
            raise TypeError("Input(s) to plot_map() must be of type " \
                          + "NavData.")

        # check for lat/lon indexes
        traj_idxs = traj_data.find_wildcard_indexes(
                    wildcards=["lat_*_deg","lon_*_deg"], max_allow=1,
                    excludes=[["lat_sigma_*_deg"],["lon_sigma_*_deg"]])

        label_name = _get_label({"":"_".join((traj_idxs["lat_*_deg"][0].split("_"))[1:-1])})

        data = {"latitude" : traj_data[traj_idxs["lat_*_deg"][0]],
                "longitude" : traj_data[traj_idxs["lon_*_deg"][0]],
                "Trajectory" : [label_name] * len(traj_data),
                }
        if sections >= 2:
            traj_data.in_rows("gps_millis")
            data["gps_millis"] = traj_data["gps_millis"]
        traj_df = pd.DataFrame.from_dict(data)
        color_discrete_map[label_name] = \
                            STANFORD_COLORS[idx % len(STANFORD_COLORS)]
        if figure_df is None:
            figure_df = traj_df
        else:
            figure_df = pd.concat([figure_df,traj_df])

    zoom, center = _zoom_center(lats=figure_df["latitude"].to_numpy(),
                                lons=figure_df["longitude"].to_numpy(),
                                width_to_height=float(0.9*width)/height)

    fig = px.scatter_mapbox(figure_df,
                            lat="latitude",
                            lon="longitude",
                            color="Trajectory",
                            color_discrete_map=color_discrete_map,
                            zoom=zoom,
                            center=center,
                            width = width,
                            height = height,
                            )
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(**kwargs)

    if sections <= 1:
        if save: # pragma: no cover
            _save_plotly(fig, titles="map", prefix=prefix, fnames=fname,
                         width=width, height=height)
        return fig

    figs = [fig]
    titles = ["map_full"]
    # break into zoom section of figures
    time_groups = np.array_split(np.sort(figure_df["gps_millis"].unique()),sections)
    for time_idx, time_group in enumerate(time_groups):
        zoomed_df = figure_df[(figure_df["gps_millis"] >= min(time_group)) \
                            & (figure_df["gps_millis"] <= max(time_group))]

        # calculate new zoom and center based on partial data
        zoom, center = _zoom_center(lats=zoomed_df["latitude"].to_numpy(),
                                    lons=zoomed_df["longitude"].to_numpy(),
                                    width_to_height=float(0.9*width)/height)
        fig = px.scatter_mapbox(figure_df,
                                lat="latitude",
                                lon="longitude",
                                color="Trajectory",
                                color_discrete_map=color_discrete_map,
                                zoom=zoom,
                                center=center,
                                width = width,
                                height = height,
                                )
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        fig.update_layout(**kwargs)

        figs.append(fig)
        titles.append("map_section_" + str(time_idx + 1))

    if save: # pragma: no cover
        _save_plotly(figs, titles=titles, prefix=prefix, fnames=fname,
                     width=width, height=height)
    return figs

def close_figures(figs=None):
    """Closes figures.

    If figs is None, then will attempt to close all matplotlib figures
    with plt.close('all')

    Parameters
    ----------
    figs : list or matplotlib.pyplot.figure or None
        List of figures or single matplotlib figure object.

    """

    if figs is None:
        plt.close('all')
    elif isinstance(figs,plt.Figure):
        plt.close(figs)
    elif isinstance(figs, list):
        for fig in figs:
            if isinstance(fig, plt.Figure):
                plt.close(fig)
    else:
        raise TypeError("Must be either a single figure or list of figures.")

def _get_new_fig(fig=None):
    """Creates new default figure and axes.

    Parameters
    ----------
    fig : matplotlib.pyplot.figure
        Previous figure to format to style.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Default NavData figure.
    axes : matplotlib.pyplot.axes
        Default NavData axes.

    """

    if fig is None:
        fig = plt.figure()
        axes = plt.gca()
    elif len(fig.get_axes()) == 0:
        axes = plt.gca()
    else:
        axes = fig.get_axes()[0]

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

    # handle units specially.
    units = {"m","km",
             "deg","rad",
             "millis","ms","sec","s","hr","min",
             "mps","kmph","mph",
             "dgps","radps",
             "mps2",
             }
    unit_replacements = {
                         "ms" : "milliseconds",
                         "millis" : "milliseconds",
                         "mps" : "m/s",
                         "kmph" : "km/hr",
                         "mph" : "miles/hr",
                         "degps" : "deg/s",
                         "radps" : "rad/s",
                         "mps2" : "m/s^2",
                        }

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

        # special exceptions for known times
        if value in ("gps_millis","unix_millis"):
            value = value.split("_")[0] + "_time_millis"

        value = value.split("_")
        if value[-1] in units:
            # make units lowercase and bracketed.
            if value[-1] in unit_replacements:
                value[-1] = unit_replacements[value[-1]]
            value = " ".join(value[:-1]).upper() + " [" + value[-1] + "]"
        else:
            value = " ".join(value).upper()

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
    figures : single or list of matplotlib.pyplot.figure objects
        Figures to be saved.
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
    if isinstance(fnames, (str, pathlib.Path)) or fnames is None:
        fnames = [fnames]

    for fig_idx, figure in enumerate(figures):

        if (len(fnames) == 1 and fnames[0] is None) \
            or fnames[fig_idx] is None:
            # create results folder if it does not yet exist.
            log_path = os.path.join(os.getcwd(),"results",fo.TIMESTAMP)
            fo.make_dir(log_path)

            # make name path friendly
            title = titles[fig_idx]
            title = title.replace(" ","_")
            title = title.replace(".","")

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

def _zoom_center(lats, lons, width_to_height = 1.25):
    """Finds optimal zoom and centering for a plotly mapbox.

    Assumed to use Mercator projection.

    Temporary solution copied from stackoverflow [1]_ and awaiting
    official implementation [2]_.

    Parameters
    --------
    lons: array-like,
        Longitude component of each location.
    lats: array-like
        Latitude component of each location.
    width_to_height: float, expected ratio of final graph's width to
        height, used to select the constrained axis.

    Returns
    -------
    zoom: float
        Plotly zoom parameter from 1 to 20.
    center: dict
        Position with 'lon' and 'lat' keys for cetner of map.

    References
    ----------
    .. [1] Richie V. https://stackoverflow.com/a/64148305/12995548.
    .. [2] https://github.com/plotly/plotly.js/issues/3434

    """

    maxlon, minlon = max(lons), min(lons)
    maxlat, minlat = max(lats), min(lats)
    center = {
        'lon': round((maxlon + minlon) / 2, 6),
        'lat': round((maxlat + minlat) / 2, 6)
    }

    # longitudinal range by zoom level (20 to 1)
    # in degrees, if centered at equator
    lon_zoom_range = np.array([
        0.0007, 0.0014, 0.003, 0.006, 0.012, 0.024, 0.048, 0.096,
        0.192, 0.3712, 0.768, 1.536, 3.072, 6.144, 11.8784, 23.7568,
        47.5136, 98.304, 190.0544, 360.0
    ])

    # assumed Mercator projection
    margin = 2.5
    height = (maxlat - minlat) * margin * width_to_height
    width = (maxlon - minlon) * margin
    lon_zoom = np.interp(width , lon_zoom_range, range(20, 0, -1))
    lat_zoom = np.interp(height, lon_zoom_range, range(20, 0, -1))
    zoom = floor(min(lon_zoom, lat_zoom))
    # zoom level higher than 18 won't load load properly as of June 2023
    zoom = min(zoom, 18)

    return zoom, center

def _save_plotly(figures, titles=None, prefix="", fnames=None,
                 width=730, height=520): # pragma: no cover
    """Saves figures to file.

    Parameters
    ----------
    figures : single or list of plotly.graph_objects.Figure objects to
        be saved.
    titles : string, path-like or list of strings
        Titles for all plots.
    prefix : string
        File prefix to add to filename.
    fnames : single or list of string or path-like
        Path to save figure to. If not None, ``fname`` is passed
        directly to plotly's write_image file parameter and prefix will
        be overwritten.
    width : int
        Figure width in pixels.
    height : int
        Figure height in pixels.

    """

    if isinstance(figures, go.Figure):
        figures = [figures]
    if isinstance(titles,str) or titles is None:
        titles = [titles]
    if isinstance(fnames, (str, pathlib.Path)) or fnames is None:
        fnames = [fnames]

    for fig_idx, figure in enumerate(figures):

        if (len(fnames) == 1 and fnames[0] is None) \
            or fnames[fig_idx] is None:
            # create results folder if it does not yet exist.
            log_path = os.path.join(os.getcwd(),"results",fo.TIMESTAMP)
            fo.make_dir(log_path)

            # make name path friendly
            title = titles[fig_idx]
            title = title.replace(" ","_")
            title = title.replace(".","")

            if prefix != "" and not prefix.endswith('_'):
                prefix += "_"
            fname = os.path.join(log_path, prefix + title \
                                                  + ".png")
        else:
            fname = fnames[fig_idx]

        while True:
            # sometimes writing a plotly image hanges for an unknown
            # reason. Hence, we call write_image in a process that is
            # automatically terminated after 180 seconds if nothing
            # happens.
            process = Process(target=_write_plotly,
                               name="write_plotly",
                               args=(figure,fname,width,height))
            process.start()

            process.join(180)
            if process.is_alive():
                process.terminate()
                process.join()
                continue
            break


def _write_plotly(figure, fname, width, height): # pragma: no cover
    """Saves figure to file.

    Automatically zooms out if plotly throws a ValueError when trying
    to zoom in too much.

    Parameters
    ----------
    figure : plotly.graph_objects.Figure
        Object to save.
    fname : string or path-like
        Path to save figure to.
    width : int
        Figure width in pixels.
    height : int
        Figure height in pixels.

    """

    while True:
        try:
            figure.write_image(fname,
                               width = width,
                               height = height,
                               )
            break
        except ValueError as error:
            figure.layout.mapbox.zoom -= 1
            if figure.layout.mapbox.zoom < 1:
                print(error)
                break
