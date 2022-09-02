"""Functions to process precise ephemerides .sp3 and .clk files.

"""

__authors__ = "Sriramya Bhamidipati"
__date__ = "09 June 2022"

import os
import warnings

from datetime import datetime, timezone
import numpy as np

from gnss_lib_py.utils.time_conversions import datetime_to_gps_millis

# Define the number of sats to create arrays for
# NUMSATS_GPS = 32
# NUMSATS_BEIDOU = 46
# NUMSATS_GLONASS = 24
# NUMSATS_GALILEO = 36
# NUMSATS_QZSS = 3

NUMSATS = {'gps': (32, 'G'),
           'galileo': (36, 'E'),
           'beidou': (46, 'C'),
           'glonass': (24, 'R'),
           'qzss': (3, 'J')}

class Sp3:
    """Class handling satellite position data (precise ephemerides)
    from .sp3 dataset.

    Notes
    -----
    (1) Not sure how to fix these pylint errors:
    precise_ephemerides.py:20:0: R0903: Too few public methods
    (0/2) (too-few-public-methods)
    """
    def __init__(self):
        self.const = None
        self.xpos = []
        self.ypos = []
        self.zpos = []
        self.tym = []
        self.utc_time = []

    def __eq__(self, other):
        """Checks if two Sp3() classes are equal to each other

        Parameters
        ----------
        other : gnss_lib_py.parsers.precise_ephemerides.Sp3
            Sp3 object that stores .sp3 parsed information
        """
        return (self.const == other.const) & \
               (self.xpos == other.xpos) & \
               (self.ypos == other.ypos) & \
               (self.zpos == other.zpos) & \
               (self.tym == other.tym) & \
               (self.utc_time == other.utc_time)

def parse_sp3(input_path, constellation = 'gps'):
    """sp3 specific loading and preprocessing for any GNSS constellation

    Parameters
    ----------
    input_path : string
        Path to sp3 file
    constellation : string (default: gps)
        Key from among {gps, galileo, glonass, beidou, qzss, etc} that
        specifies which GNSS constellation to be parsed from .sp3 file

    Returns
    -------
    sp3data : Sp3
        Array of Sp3 classes where each corresponds to a satellite with the
        specified constellation and is populated with parsed sp3 information

    Notes
    -----
    This parser function does not process all available GNSS constellations
    at once, i.e., needs to be independently called for each desired one

    0th array of the Clk class is always empty since PRN=0 does not exist

    Based on code written by J. Makela.
    AE 456, Global Navigation Sat Systems, University of Illinois
    Urbana-Champaign. Fall 2015

    TOCLARIFY:
    (1) Should the other GPS ephemeris python file be called
    broadcast_ephemeris for consistency and clarity?
    (2) Should there be a history sub-heading in function descriptions
    as well, for better clarity on when was that function last updated?
    (3) Not sure how to fix this pylint error:
    precise_ephemerides.py:74:37: W1514: Using open without explicitly
    specifying an encoding (unspecified-encoding)
    precise_ephemerides.py:38:0: R0912: Too many branches (14/12)
    (too-many-branches) -Ans: list format/ignore.
    (4) Do we want automatic downloading of relevant .sp3 and .clk files,
    similar to what EphemerisManager does? --> task for later

    References
    ----------
    .. [1]  https://files.igs.org/pub/data/format/sp3d.pdf
            Accessed as of August 20, 2022
    """
    # Initial checks for loading sp3_path
    if not isinstance(input_path, str):
        raise TypeError("input_path must be string")
    if not os.path.exists(input_path):
        raise OSError("file not found")

    # Load in the file
    with open(input_path, 'r', encoding="utf-8") as infile:
        data = [line.strip() for line in infile]

#     data = [line.strip() for line in open(input_path)]

    # Poll the total no. of satellites based on constellation specified
    if constellation in NUMSATS.keys():
        nsvs = NUMSATS[constellation][0]
    else:
        raise RuntimeError("No support exists for specified constellation")

    # Create a sp3 class for each expected satellite
    sp3data = []
    for _ in np.arange(0, nsvs+1):
        sp3data.append(Sp3())
        sp3data[-1].const = constellation

    # Loop through each line
    for dval in data:
        if len(dval) == 0:
            # No data
            continue

        if dval[0] == '*':
            # A new record
            # Get the date
            temp = dval.split()
            curr_time = datetime( int(temp[1]), int(temp[2]), \
                                  int(temp[3]), int(temp[4]), \
                                  int(temp[5]),int(float(temp[6])), \
                                  tzinfo=timezone.utc )
            gps_millis = datetime_to_gps_millis(curr_time, add_leap_secs = False)

        if 'P' in dval[0]:
            # A satellite record.  Get the satellite number, and coordinate (X,Y,Z) info
            temp = dval.split()

            if temp[0][1] == NUMSATS[constellation][1]:
                prn = int(temp[0][2:])
                sp3data[prn].utc_time.append(curr_time)
                sp3data[prn].tym.append(gps_millis)
                sp3data[prn].xpos.append(float(temp[1])*1e3)
                sp3data[prn].ypos.append(float(temp[2])*1e3)
                sp3data[prn].zpos.append(float(temp[3])*1e3)

    # Add warning in case any satellite PRN does not have data
    no_data_arrays = []
    for prn in np.arange(1, nsvs+1):
        if len(sp3data[prn].tym) == 0:
            no_data_arrays.append(prn)
    if len(no_data_arrays) == nsvs:
        warnings.warn("No sp3 data found for PRNs: "+str(no_data_arrays), RuntimeWarning)

    return sp3data

class Clk:
    """Class handling biases in satellite clock (precise ephemerides)
    from .clk dataset.

    Notes
    -----
    (1) Not sure how to fix these pylint errors:
    precise_ephemerides.py:126:0: R0903: Too few public methods
    (0/2) (too-few-public-methods)
    """
    def __init__(self):
        self.const = None
        self.clk_bias = []
        self.utc_time = []
        self.tym = []

    def __eq__(self, other):
        """Checks if two Clk() classes are equal to each other

        Parameters
        ----------
        other : gnss_lib_py.parsers.precise_ephemerides.Clk
            Clk object that stores .clk parsed information
        """
        return (self.const == other.const) & \
               (self.clk_bias == other.clk_bias) & \
               (self.tym == other.tym) & \
               (self.utc_time == other.utc_time)

def parse_clockfile(input_path, constellation = 'gps'):
    """Clk specific loading and preprocessing for any GNSS constellation

    Parameters
    ----------
    input_path : string
        Path to clk file
    constellation : string (default: gps)
        Key from among {gps, galileo, glonass, beidou, qzss, etc} that
        specifies which GNSS constellation to be parsed from .sp3 file

    Returns
    -------
    clkdata : Clk
        Array of Clk classes where each corresponds to a satellite with the
        specified constellation and is populated with parsed clk information

    Notes
    -----
    This parser function does not process all available GNSS constellations
    at once, i.e., needs to be independently called for each desired one

    0th array of the Clk class is always empty since PRN=0 does not exist

    Based on code written by J. Makela.
    AE 456, Global Navigation Sat Systems, University of Illinois
    Urbana-Champaign. Fall 2015

    (1) Not sure how to fix the pylint error:
    precise_ephemerides.py:197:11: W1514: Using open without explicitly
    specifying an encoding (unspecified-encoding)
    precise_ephemerides.py:156:0: R0912: Too many branches (19/12)
    (too-many-branches)
    Maybe this link: https://peps.python.org/pep-0597/#id11

    References
    -----
    .. [1]  https://files.igs.org/pub/data/format/rinex_clock300.txt
            Accessed as of August 24, 2022
    """

    # Initial checks for loading sp3_path
    if not isinstance(input_path, str):
        raise TypeError("input_path must be string")
    if not os.path.exists(input_path):
        raise OSError("file not found")

    # Poll the total no. of satellites based on constellation specified
    if constellation in NUMSATS.keys():
        nsvs = NUMSATS[constellation][0]
    else:
        raise RuntimeError("No support exists for specified constellation")

    # Create a CLK class for each expected satellite
    clkdata = []
    for _ in np.arange(0, nsvs+1):
        clkdata.append(Clk())
        clkdata[-1].const = constellation

    # Read Clock file
    with open(input_path, 'r', encoding="utf-8") as infile:
        clk = infile.readlines()

    line = 0
    while True:
        if 'OF SOLN SATS' not in clk[line]:
            del clk[line]
        else:
            line +=1
            break

    line = 0
    while True:
        if 'END OF HEADER' not in clk[line]:
            line +=1
        else:
            del clk[0:line+1]
            break

    timelist = []
    for _, clk_val in enumerate(clk):
        if clk_val[0:2]=='AS':
            timelist.append(clk_val.split())

    for _, timelist_val in enumerate(timelist):
        dval = timelist_val[1]

        if dval[0] == NUMSATS[constellation][1]:
            prn = int(dval[1:])
            curr_time = datetime(year = int(timelist_val[2]), \
                                 month = int(timelist_val[3]), \
                                 day = int(timelist_val[4]), \
                                 hour = int(timelist_val[5]), \
                                 minute = int(timelist_val[6]), \
                                 second = int(float(timelist_val[7])), \
                                 tzinfo=timezone.utc)
            clkdata[prn].utc_time.append(curr_time)
            gps_millis = datetime_to_gps_millis(curr_time, add_leap_secs = False)
            clkdata[prn].tym.append(gps_millis)
            clkdata[prn].clk_bias.append(float(timelist_val[9]))

    infile.close() # close the file

    # Add warning in case any satellite PRN does not have data
    no_data_arrays = []
    for prn in np.arange(1, nsvs+1):
        if len(clkdata[prn].tym) == 0:
            no_data_arrays.append(prn)
    if len(no_data_arrays) == nsvs:
        warnings.warn("No clk data found for PRNs: " + str(no_data_arrays), RuntimeWarning)

    return clkdata
