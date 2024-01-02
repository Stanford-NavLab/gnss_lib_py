"""Tests for visualizations.

"""

__authors__ = "D. Knowles"
__date__ = "22 Jun 2022"

import random

import pytest
import numpy as np

import gnss_lib_py.visualizations.style as style

# pylint: disable=protected-access

def testget_label():
    """Test for getting nice labels.

    """

    assert style.get_label({"signal_type" : "l1"}) == "L1"
    assert style.get_label({"signal_type" : "g1"}) == "G1"
    assert style.get_label({"signal_type" : "b1i"}) == "B1i"
    # shouldn't do lowercase 'i' trick if not signal_type
    assert style.get_label({"random" : "BDS_B1I"}) == "BDS B1I"

    assert style.get_label({"gnss_id" : "beidou"}) == "BeiDou"
    assert style.get_label({"gnss_id" : "gps"}) == "GPS"
    assert style.get_label({"gnss_id" : "galileo"}) == "Galileo"

    assert style.get_label({"gnss_id" : "galileo",
                           "signal_type" : "b1i"}) == "Galileo B1i"

    assert style.get_label({"row" : "x_rx_m"}) == "X RX [m]"
    assert style.get_label({"row" : "lat_rx_deg"}) == "LAT RX [deg]"
    assert style.get_label({"row" : "vx_sv_mps"}) == "VX SV [m/s]"

    with pytest.raises(TypeError) as excinfo:
        style.get_label(["should","fail"])
    assert "dictionary" in str(excinfo.value)

def testsort_gnss_ids():
    """Test sorting GNSS IDs.

    """

    unsorted_ids = ["galileo","beta","beidou","irnss","gps","unknown","glonass",
                "alpha","qzss","sbas"]
    sorted_ids = ["gps","glonass","galileo","beidou","qzss","irnss","sbas",
                  "unknown", "alpha", "beta"]

    assert style.sort_gnss_ids(unsorted_ids) == sorted_ids
    assert style.sort_gnss_ids(np.array(unsorted_ids)) == sorted_ids
    assert style.sort_gnss_ids(set(unsorted_ids)) == sorted_ids
    assert style.sort_gnss_ids(tuple(unsorted_ids)) == sorted_ids

    for _ in range(100):
        random.shuffle(unsorted_ids)
        assert style.sort_gnss_ids(unsorted_ids) == sorted_ids
        assert style.sort_gnss_ids(np.array(unsorted_ids)) == sorted_ids
        assert style.sort_gnss_ids(set(unsorted_ids)) == sorted_ids
        assert style.sort_gnss_ids(tuple(unsorted_ids)) == sorted_ids

def test_close_figures_fail():
    """Test expected fail conditions.

    """

    style.close_figures([])

    with pytest.raises(TypeError) as excinfo:
        style.close_figures(0.)
    assert "figure" in str(excinfo.value)
