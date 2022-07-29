"""Simple file operation utilities.

"""

__authors__ = "D. Knowles"
__date__ = "01 Jul 2022"

import os
import time

import matplotlib.pyplot as plt

def make_dir(directory): # pragma: no cover
    """Create a file directory if it doesn't yet exist.

    Parameters
    ----------
    directory : string
        Filepath of directory to create if it does not exist.

    """

    # create directory if it doesn't yet exist
    if not os.path.isdir(directory):
        try:
            os.makedirs(directory)
        except OSError as error:
            raise OSError("Unable to create directory " + directory) from error

def save_figure(fig, filename): # pragma: no cover
    """Saves figure as a png.

    Parameters
    ----------
    fig : matplotlib.pyplot.figure
        Figure to save.
    filename : string
        Filename for figure.

    """
    fig.savefig(filename,
                dpi=300.,
                format="png",
                bbox_inches="tight")


def close_figures(figs): # pragma: no cover
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

def get_timestamp():
    """Returns timestamp of the current time.

    Returns
    -------
    timestamp : string
        Timestamp in order of year, month, day, hour, minute, second
        without spaces or puncuation

    """
    timestamp =  time.strftime("%Y%m%d%H%M%S")
    return timestamp

def get_lib_dir():
    """Returns filepath to the Pixel4XL data

    Returns
    -------
    filepath : string
        Filepath to the main gnss_lib_py repository directory

    """
    return os.path.dirname(os.path.dirname(os.path.dirname(
                           os.path.realpath(__file__))))
