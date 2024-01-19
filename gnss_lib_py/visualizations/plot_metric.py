"""Visualization functions for GNSS data.

"""

__authors__ = "D. Knowles"
__date__ = "27 Jan 2022"

import numpy as np
import matplotlib.pyplot as plt

from gnss_lib_py.visualizations.style import *
from gnss_lib_py.navdata.navdata import NavData

def plot_metric(navdata, *args, groupby=None, avg_y=False, fig=None,
                title=None, save=False, prefix="", fname=None,
                markeredgecolor="k", markeredgewidth=0.2, **kwargs):
    """Plot specific metric from a row of the NavData class.

    Parameters
    ----------
    navdata : gnss_lib_py.navdata.navdata.NavData
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
            title = get_label({y_metric:y_metric})
    else:
        if title is None:
            title = get_label({y_metric:y_metric}) + " vs. " \
                  + get_label({x_metric:x_metric})
        xlabel = get_label({x_metric:x_metric})

    if groupby is not None:
        all_groups = np.unique(navdata[groupby])
        if groupby == "gnss_id":
            all_groups = sort_gnss_ids(all_groups)
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
                      label=get_label({groupby:group}),
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
                   title=get_label({groupby:groupby}))

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(get_label({y_metric:y_metric}))
    fig.set_layout_engine(layout="tight")

    if save: # pragma: no cover
        save_figure(fig, title, prefix, fname)
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
    navdata : gnss_lib_py.navdata.navdata.NavData
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
    for constellation in sort_gnss_ids(np.unique(navdata["gnss_id"])):
        const_subset = navdata.where("gnss_id",constellation)

        if "signal_type" in const_subset.rows:
            for signal in np.unique(const_subset["signal_type"]):
                title = get_label({"gnss_id":constellation,"signal_type":signal})
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
            title = get_label({"gnss_id":constellation})
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

def _parse_metric_args(navdata, *args):
    """Parses arguments and raises error if metrics are nonnumeric.

    Parameters
    ----------
    navdata : gnss_lib_py.navdata.navdata.NavData
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
