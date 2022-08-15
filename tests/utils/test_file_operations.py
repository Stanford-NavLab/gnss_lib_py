"""Test file operation utilities.

"""

__authors__ = "D. Knowles"
__date__ = "01 Jul 2022"

import os

import gnss_lib_py.utils.file_operations as fo

# test_make_dir
# test save_figure
# test close_figures

def test_get_timestamp():
    """Test for getting timestamp.

    """
    timestamp = fo.get_timestamp()

    # timestamp should be of length 14, YYYYMMDDHHMMSS
    assert len(timestamp) == 14

    # timestamp should all be numeric characters
    assert timestamp.isnumeric()
