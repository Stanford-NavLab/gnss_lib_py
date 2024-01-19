"""Visualization functions for GNSS data.

"""

__authors__ = "D. Knowles"
__date__ = "27 Jan 2022"

import os
import pathlib

import numpy as np
from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import gnss_lib_py.utils.file_operations as fo

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

def get_label(inputs):
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
        raise TypeError("get_label input must be dictionary.")

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

def sort_gnss_ids(unsorted_gnss_ids):
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

def save_figure(figures, titles=None, prefix="", fnames=None): # pragma: no cover
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
