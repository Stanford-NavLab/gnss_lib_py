"""Test file operation utilities.

"""

__authors__ = "D. Knowles"
__date__ = "01 Jul 2022"

import gnss_lib_py.utils.file_operations as fo

# test_mkdir
# test save_figure
# test close_figures

def test_get_timestamp():
    timestamp = fo.get_timestamp()

    assert len(timestamp) == 14

    assert timestamp.isnumeric()
