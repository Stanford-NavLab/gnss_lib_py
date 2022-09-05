"""Simple file operation utilities.

"""

__authors__ = "D. Knowles"
__date__ = "01 Jul 2022"

import os
import time

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

def print_directory_levels():
    """Prints out file directory levels.

    """

    printed_path = os.path.realpath(__file__)

    for level in range(5):
        print(f"{level} level(s) up: {printed_path}")
        printed_path = os.path.dirname(printed_path)

def print_cwd():
    """Prints out file directory levels.

    """

    print("cwd:",os.getcwd())
    print("dir above cwd:",os.path.dirname(os.getcwd()))
