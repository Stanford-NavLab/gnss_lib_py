"""Tests for visualizations.

"""

__authors__ = "D. Knowles"
__date__ = "22 Jun 2022"

import os
import random

import pytest
import numpy as np
import plotly.graph_objects as go
from pytest_lazyfixture import lazy_fixture
import matplotlib as mpl
import matplotlib.pyplot as plt

import gnss_lib_py.utils.visualizations as viz
from gnss_lib_py.algorithms.snapshot import solve_wls
from gnss_lib_py.utils.coordinates import geodetic_to_ecef

# pylint: disable=protected-access
