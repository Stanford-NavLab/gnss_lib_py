"""Test file operation utilities.

"""

__authors__ = "D. Knowles"
__date__ = "01 Jul 2022"

import gnss_lib_py.utils.file_operations as fo

# test_make_dir

def test_get_timestamp():
    """Test for getting timestamp.

    """
    timestamp = fo.get_timestamp()

    # timestamp should be of length 14, YYYYMMDDHHMMSS
    assert len(timestamp) == 14

    # timestamp should all be numeric characters
    assert timestamp.isnumeric()

def test_directory_printing():
    """Test directory printing functions.
    
    """
    fo.print_directory_levels()
    fo.print_cwd()
