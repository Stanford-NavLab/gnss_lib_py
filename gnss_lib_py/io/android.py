########################################################################
# Author(s):    Shubh Gupta, Adam Dai
# Date:         13 Jul 2021
# Desc:         Functions to process Android measurements
########################################################################

import os
import sys
# append <path>/gnss_lib_py/gnss_lib_py/ to path
sys.path.append(os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__))))
import csv
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from core.constants import GPSConsts


def extract_timedata(input_path):
    """Extracts raw and fix data from GNSS log file.

    Parameters
    ----------
    input_path : string
        File location of data file to read.

    Returns
    -------
    header_raw : string
        Header for raw data.
    raw_data : list
        Raw data appended line by line.
    header_fix : string
        Header for fix data.
    fix_data : list
        Fix data appended line by line.

    Notes
    -----
    This function doesn't appear to be used anywhere, is it needed?

    """
    raw_data = []
    fix_data = []
    header_fix = ''
    header_raw = ''
    with open(input_path, 'r') as f:
        t = -1
        for line in f:
            if line[0] == '#':
                line_data = line[2:].rstrip('\n').replace(" ","").split(",")
                if line_data[0] == 'Raw':
                    header_raw = line_data[1:]
                elif line_data[0] == 'Fix':
                    header_fix = line_data[1:]
                continue
            line_data = line.rstrip('\n').replace(" ","").split(",")
            if line_data[0] == 'Fix':
                fix_data.append(line_data[1:])
                raw_data.append([])
                t += 1
            elif line_data[0] == 'Raw':
                raw_data[t].append(line_data[1:])
    return header_raw, raw_data, header_fix, fix_data

def make_csv(input_path, field):
    """Write specific data types from a GNSS android log to a CSV.

    Parameters
    ----------
    input_path : string
        File location of data file to read.
    field : string
        Type of data to extract. Valid options are either "Raw",
        "Accel", or "Gyro".

    Returns
    -------
    out_path : string
        New file location of the exported CSV.

    Notes
    -----
    Based off of MATLAB code from Google's gps-measurement-tools
    repository: https://github.com/google/gps-measurement-tools. Compare
    with MakeCsv() in opensource/ReadGnssLogger.m

    """
    out_path = field + ".csv"
    with open(out_path, 'w') as f:
        writer = csv.writer(f)
        with open(input_path, 'r') as f:
            for line in f:
                # Comments in the log file
                if line[0] == '#':
                    # Remove initial '#', spaces, trailing newline and split using commas as delimiter
                    line_data = line[2:].rstrip('\n').replace(" ","").split(",")
                    if line_data[0] == field:
                        writer.writerow(line_data[1:])
                # Data in file
                else:
                    # Remove spaces, trailing newline and split using commas as delimiter
                    line_data = line.rstrip('\n').replace(" ","").split(",")
                    if line_data[0] == field:
                        writer.writerow(line_data[1:])
    return out_path


def make_gnss_dataframe(input_path, verbose=False):
    """Read Android raw file and produce gnss dataframe objects

    Parameters
    ----------
    input_path : string
        File location of data file to read.
    verbose : bool
        If true, will print out any problems that were detected.

    Returns
    -------
    corrected_measurements : pandas dataframe
        Dataframe that contains a corrected version of the measurements.
    andorid fixes : pandas dataframe
        Dataframe that contains the andorid fixes from the log file.

    """
    with open(input_path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0][0] == '#':
                if 'Fix' in row[0]:
                    android_fixes = [row[1:]]
                elif 'Raw' in row[0]:
                    measurements = [row[1:]]
            else:
                if row[0] == 'Fix':
                    android_fixes.append(row[1:])
                elif row[0] == 'Raw':
                    measurements.append(row[1:])

    android_fixes = pd.DataFrame(android_fixes[1:], columns = android_fixes[0])
    measurements = pd.DataFrame(measurements[1:], columns = measurements[0])

    corrected_measurements = correct_log(measurements, verbose=verbose)

    return corrected_measurements, android_fixes


def make_imu_dataframe(input_path, verbose=False):
    """Read Android raw file and produce IMU dataframe objects

    Parameters
    ----------
    input_path : string
        File location of data file to read.
    verbose : bool
        If true, will print out any problems that were detected.

    Returns
    -------
    accel : pandas dataframe
        Dataframe that contains the accel measurements from the log.
    gyro : pandas dataframe
        Dataframe that contains the gyro measurements from the log.

    """
    with open(input_path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0][0] == '#':
                if 'Accel' in row[0]:
                    accel = [row[1:]]
                elif 'Gyro' in row[0]:
                    gyro = [row[1:]]
            else:
                if row[0] == 'Accel':
                    accel.append(row[1:])
                elif row[0] == 'Gyro':
                    gyro.append(row[1:])

    accel = pd.DataFrame(accel[1:], columns = accel[0])
    gyro = pd.DataFrame(gyro[1:], columns = gyro[0])

    return accel, gyro

#
def correct_log(measurements, verbose=False, carrier_phase_checks = False):
    """Compute required quantities from the log and check for errors.

    This is a master function that calls the other correction functions
    to validate a GNSS measurement log dataframe.

    Parameters
    ----------
    measurements : pandas dataframe
        pandas dataframe that holds gnss meassurements
    verbose : bool
        If true, will print out any problems that were detected.
    carrier_phase_checks : bool
        If true, completes carrier phase checks

    Returns
    -------
    measurements : pandas dataframe
        same dataframe with possibly some fixes or column additions

    """
    # Add leading 0
    measurements.loc[measurements['Svid'].str.len() == 1, 'Svid'] = '0' + measurements['Svid']

    # Compatibility with RINEX files
    measurements.loc[measurements['ConstellationType'] == '1', 'Constellation'] = 'G'
    measurements.loc[measurements['ConstellationType'] == '3', 'Constellation'] = 'R'
    measurements['SvName'] = measurements['Constellation'] + measurements['Svid']

    # Drop non-GPS measurements
    measurements = measurements.loc[measurements['Constellation'] == 'G']

    # TODO: Measurements should be discarded if the constellation is unknown.

    # Convert columns to numeric representation
    # assign() function prevents SettingWithCopyWarning
    measurements = measurements.assign(Cn0DbHz=pd.to_numeric(measurements['Cn0DbHz']))
    measurements = measurements.assign(TimeNanos=pd.to_numeric(measurements['TimeNanos']))
    measurements = measurements.assign(FullBiasNanos=pd.to_numeric(measurements['FullBiasNanos']))
    measurements = measurements.assign(ReceivedSvTimeNanos=pd.to_numeric(measurements['ReceivedSvTimeNanos']))
    measurements = measurements.assign(PseudorangeRateMetersPerSecond=pd.to_numeric(measurements['PseudorangeRateMetersPerSecond']))
    measurements = measurements.assign(ReceivedSvTimeUncertaintyNanos=pd.to_numeric(measurements['ReceivedSvTimeUncertaintyNanos']))

    # Check clock fields
    error_logs = []
    measurements, error_logs = check_gnss_clock(measurements, error_logs)
    measurements, error_logs = check_gnss_measurements(measurements, error_logs)
    if carrier_phase_checks:
        measurements, error_logs = check_carrier_phase(measurements, error_logs)
    measurements, error_logs = compute_times(measurements, error_logs)
    measurements, error_logs = compute_pseudorange(measurements, error_logs)

    if verbose:
        if len(error_logs)>0:
            print("Following problems detected:")
            print(error_logs)
        else:
            print("No problems detected.")
    return measurements

def check_gnss_clock(gnssRaw, gnssAnalysis):
    """Checks and fixes clock field errors

    Additonal checks added from [1]_.

    Parameters
    ----------
    gnssRaw : pandas dataframe
        pandas dataframe that holds gnss meassurements
    gnssAnalysis : list
        holds any error messages

    Returns
    -------
    gnssRaw : pandas dataframe
        same dataframe with possibly some fixes or column additions
    gnssAnalysis : list
        holds any error messages

    Notes
    -----
    Based off of MATLAB code from Google's gps-measurement-tools
    repository: https://github.com/google/gps-measurement-tools. Compare
    with CheckGnssClock() in opensource/ReadGnssLogger.m

    References
    ----------
    .. [1] Fu, Guoyu Michael, Mohammed Khider, and Frank van Diggelen.
       "Android Raw GNSS Measurement Datasets for Precise Positioning."
       Proceedings of the 33rd International Technical Meeting of the
       Satellite Division of The Institute of Navigation (ION GNSS+
       2020). 2020.

    """
    # list of clock fields
    gnssClockFields = [
      'TimeNanos',
      'TimeUncertaintyNanos',
      'TimeOffsetNanos',
      'LeapSecond',
      'FullBiasNanos',
      'BiasUncertaintyNanos',
      'DriftNanosPerSecond',
      'DriftUncertaintyNanosPerSecond',
      'HardwareClockDiscontinuityCount',
      'BiasNanos'
      ]
    for field in gnssClockFields:
        if field not in gnssRaw.head():
            gnssAnalysis.append('WARNING: '+field+' (Clock) is missing from GNSS Logger file')
        else:
            gnssRaw.loc[:,field] = pd.to_numeric(gnssRaw[field])
    ok = all(x in gnssRaw.head() for x in ['TimeNanos', 'FullBiasNanos'])
    if not ok:
        gnssAnalysis.append('FAIL Clock check')
        return gnssRaw, gnssAnalysis

    # Measurements should be discarded if TimeNanos is empty
    if gnssRaw["TimeNanos"].isnull().values.any():
        gnssRaw.dropna(how = "any", subset = ["TimeNanos"],
                       inplace = True)
        gnssAnalysis.append('empty or invalid TimeNanos')

    if 'BiasNanos' not in gnssRaw.head():
        gnssRaw.loc[:,'BiasNanos'] = 0
    if 'TimeOffsetNanos' not in gnssRaw.head():
        gnssRaw.loc[:,'TimeOffsetNanos'] = 0
    if 'HardwareClockDiscontinuityCount' not in gnssRaw.head():
        gnssRaw.loc[:,'HardwareClockDiscontinuityCount'] = 0
        gnssAnalysis.append('WARNING: Added HardwareClockDiscontinuityCount=0 because it is missing from GNSS Logger file')

    # measurements should be discarded if FullBiasNanos is zero or invalid
    if any(gnssRaw.FullBiasNanos >= 0):
        gnssRaw.FullBiasNanos = -1*gnssRaw.FullBiasNanos
        gnssAnalysis.append('WARNING: FullBiasNanos wrong sign. Should be negative. Auto changing inside check_gnss_clock')

    # Measurements should be discarded if BiasUncertaintyNanos is too
    # large ## TODO: figure out how to choose this parameter better
    if any(gnssRaw.BiasUncertaintyNanos >= 40.):
        count = (gnssRaw["BiasUncertaintyNanos"] >= 40.).sum()
        gnssAnalysis.append(str(count) +
         ' rows with too large BiasUncertaintyNanos')
        gnssRaw = gnssRaw[gnssRaw["BiasUncertaintyNanos"] < 40.]


    gnssRaw = gnssRaw.assign(allRxMillis = ((gnssRaw.TimeNanos - gnssRaw.FullBiasNanos)/1e6))
    # gnssRaw['allRxMillis'] = ((gnssRaw.TimeNanos - gnssRaw.FullBiasNanos)/1e6)
    return gnssRaw, gnssAnalysis


def check_gnss_measurements(gnssRaw, gnssAnalysis):
    """Checks that GNSS measurement fields exist in dataframe.

    Additonal checks added from [2]_.

    Parameters
    ----------
    gnssRaw : pandas dataframe
        pandas dataframe that holds gnss meassurements
    gnssAnalysis : list
        holds any error messages

    Returns
    -------
    gnssRaw : pandas dataframe
        exact same dataframe as input (why is this a return?)
    gnssAnalysis : list
        holds any error messages

    Notes
    -----
    Based off of MATLAB code from Google's gps-measurement-tools
    repository: https://github.com/google/gps-measurement-tools. Compare
    with ReportMissingFields() in opensource/ReadGnssLogger.m

    References
    ----------
    .. [2] Fu, Guoyu Michael, Mohammed Khider, and Frank van Diggelen.
       "Android Raw GNSS Measurement Datasets for Precise Positioning."
       Proceedings of the 33rd International Technical Meeting of the
       Satellite Division of The Institute of Navigation (ION GNSS+
       2020). 2020.

    """
    # list of measurement fields
    gnssMeasurementFields = [
        'Cn0DbHz',
        'ConstellationType',
        'MultipathIndicator',
        'PseudorangeRateMetersPerSecond',
        'PseudorangeRateUncertaintyMetersPerSecond',
        'ReceivedSvTimeNanos',
        'ReceivedSvTimeUncertaintyNanos',
        'State',
        'Svid',
        'AccumulatedDeltaRangeMeters',
        'AccumulatedDeltaRangeUncertaintyMeters'
        ]
    for field in gnssMeasurementFields:
        if field not in gnssRaw.head():
            gnssAnalysis.append('WARNING: '+field+' (Measurement) is missing from GNSS Logger file')

    # measurements should be discarded if state is neither
    # STATE_TOW_DECODED nor STATE_TOW_KNOWN
    gnssRaw = gnssRaw.assign(State=pd.to_numeric(gnssRaw['State']))
    STATE_TOW_DECODED = 0x8
    STATE_TOW_KNOWN = 0x4000
    invalid_state_count = np.invert((gnssRaw["State"] & STATE_TOW_DECODED).astype(bool) |
                              (gnssRaw["State"] & STATE_TOW_KNOWN).astype(bool)).sum()
    if invalid_state_count > 0:
        gnssAnalysis.append(str(invalid_state_count) + " rows have " + \
                            "state TOW neither decoded nor known")
        gnssRaw = gnssRaw[(gnssRaw["State"] & STATE_TOW_DECODED).astype(bool) |
                          (gnssRaw["State"] & STATE_TOW_KNOWN).astype(bool)]

    # Measurements should be discarded if ReceivedSvTimeUncertaintyNanos
    # is high ## TODO: figure out how to choose this parameter better
    if any(gnssRaw.ReceivedSvTimeUncertaintyNanos >= 150.):
        count = (gnssRaw["ReceivedSvTimeUncertaintyNanos"] >= 150.).sum()
        gnssAnalysis.append(str(count) +
         ' rows with too large ReceivedSvTimeUncertaintyNanos')
        gnssRaw = gnssRaw[gnssRaw["ReceivedSvTimeUncertaintyNanos"] < 150.]

    # convert multipath indicator to numeric
    gnssRaw = gnssRaw.assign(MultipathIndicator=pd.to_numeric(gnssRaw['MultipathIndicator']))

    return gnssRaw, gnssAnalysis

def check_carrier_phase(gnssRaw, gnssAnalysis):
    """Checks that carrier phase measurements.

    Checks taken from [3]_.

    Parameters
    ----------
    gnssRaw : pandas dataframe
        pandas dataframe that holds gnss meassurements
    gnssAnalysis : list
        holds any error messages

    Returns
    -------
    gnssRaw : pandas dataframe
        exact same dataframe as input (why is this a return?)
    gnssAnalysis : list
        holds any error messages

    References
    ----------
    .. [3] Fu, Guoyu Michael, Mohammed Khider, and Frank van Diggelen.
       "Android Raw GNSS Measurement Datasets for Precise Positioning."
       Proceedings of the 33rd International Technical Meeting of the
       Satellite Division of The Institute of Navigation (ION GNSS+
       2020). 2020.

    """

    # Measurements should be discarded if AdrState violates
    # ADR_STATE_VALID == 1 & ADR_STATE_RESET == 0
    # & ADR_STATE_CYCLE_SLIP == 0
    gnssRaw = gnssRaw.assign(AccumulatedDeltaRangeState=pd.to_numeric(gnssRaw['AccumulatedDeltaRangeState']))
    ADR_STATE_VALID = 0x1
    ADR_STATE_RESET = 0x2
    ADR_STATE_CYCLE_SLIP = 0x4

    invalid_state_count = np.invert((gnssRaw["AccumulatedDeltaRangeState"] & ADR_STATE_VALID).astype(bool) &
                          np.invert((gnssRaw["AccumulatedDeltaRangeState"] & ADR_STATE_RESET).astype(bool)) &
                          np.invert((gnssRaw["AccumulatedDeltaRangeState"] & ADR_STATE_CYCLE_SLIP).astype(bool))).sum()
    if invalid_state_count > 0:
        gnssAnalysis.append(str(invalid_state_count) + " rows have " + \
                            "ADRstate invalid")
        gnssRaw = gnssRaw[(gnssRaw["AccumulatedDeltaRangeState"] & ADR_STATE_VALID).astype(bool) &
                          np.invert((gnssRaw["AccumulatedDeltaRangeState"] & ADR_STATE_RESET).astype(bool)) &
                          np.invert((gnssRaw["AccumulatedDeltaRangeState"] & ADR_STATE_CYCLE_SLIP).astype(bool))]

    # Measurements should be discarded if AccumulatedDeltaRangeUncertaintyMeters
    # is too large ## TODO: figure out how to choose this parameter better
    gnssRaw = gnssRaw.assign(AccumulatedDeltaRangeUncertaintyMeters=pd.to_numeric(gnssRaw['AccumulatedDeltaRangeUncertaintyMeters']))
    if any(gnssRaw.AccumulatedDeltaRangeUncertaintyMeters >= 0.15):
        count = (gnssRaw["AccumulatedDeltaRangeUncertaintyMeters"] >= 0.15).sum()
        gnssAnalysis.append(str(count) +
         ' rows with too large AccumulatedDeltaRangeUncertaintyMeters')
        gnssRaw = gnssRaw[gnssRaw["AccumulatedDeltaRangeUncertaintyMeters"] < 0.15]

    return gnssRaw, gnssAnalysis


def compute_times(gnssRaw, gnssAnalysis):
    """Compute times and epochs for GNSS measurements.

    Additional checks added from [4]_.

    Parameters
    ----------
    gnssRaw : pandas dataframe
        pandas dataframe that holds gnss meassurements
    gnssAnalysis : list
        holds any error messages

    Returns
    -------
    gnssRaw : pandas dataframe
        Dataframe with added columns updated.
    gnssAnalysis : list
        Holds any error messages. This function doesn't actually add any
        error messages, but it is a nice thought.

    Notes
    -----
    Based off of MATLAB code from Google's gps-measurement-tools
    repository: https://github.com/google/gps-measurement-tools. Compare
    with opensource/ProcessGnssMeas.m

    References
    ----------
    .. [4] Fu, Guoyu Michael, Mohammed Khider, and Frank van Diggelen.
       "Android Raw GNSS Measurement Datasets for Precise Positioning."
       Proceedings of the 33rd International Technical Meeting of the
       Satellite Division of The Institute of Navigation (ION GNSS+
       2020). 2020.

    """
    gpsepoch = datetime(1980, 1, 6, 0, 0, 0)
    WEEKSEC = 604800
    gnssRaw['GpsWeekNumber'] = np.floor(-1*gnssRaw['FullBiasNanos']*1e-9/WEEKSEC)
    gnssRaw['GpsTimeNanos'] = gnssRaw['TimeNanos'] - (gnssRaw['FullBiasNanos'] + gnssRaw['BiasNanos'])

    # Measurements should be discarded if arrival time is negative
    if sum(gnssRaw['GpsTimeNanos'] <= 0) > 0:
        gnssRaw = gnssRaw[gnssRaw['GpsTimeNanos'] > 0]
        gnssAnalysis.append("negative arrival times removed")
    # TODO: Measurements should be discarded if arrival time is
    # unrealistically large

    gnssRaw['tRxNanos'] = (gnssRaw['TimeNanos']+gnssRaw['TimeOffsetNanos'])-(gnssRaw['FullBiasNanos'].iloc[0]+gnssRaw['BiasNanos'].iloc[0])
    gnssRaw['tRxSeconds'] = 1e-9*gnssRaw['tRxNanos'] - WEEKSEC * gnssRaw['GpsWeekNumber']
    gnssRaw['tTxSeconds'] = 1e-9*(gnssRaw['ReceivedSvTimeNanos'] + gnssRaw['TimeOffsetNanos'])
    gnssRaw['LeapSecond'] = gnssRaw['LeapSecond'].fillna(0)
    gnssRaw["UtcTimeNanos"] = pd.to_datetime(gnssRaw['GpsTimeNanos']  - gnssRaw["LeapSecond"] * 1E9, utc = True, origin=gpsepoch)
    gnssRaw['UnixTime'] = pd.to_datetime(gnssRaw['GpsTimeNanos'], utc = True, origin=gpsepoch)
    # TODO: Check if UnixTime is the same as UtcTime, if so remove it

    gnssRaw['Epoch'] = 0
    gnssRaw.loc[gnssRaw['UnixTime'] - gnssRaw['UnixTime'].shift() > timedelta(milliseconds=200), 'Epoch'] = 1
    gnssRaw['Epoch'] = gnssRaw['Epoch'].cumsum()
    return gnssRaw, gnssAnalysis

def compute_pseudorange(gnssRaw, gnssAnalysis):
    """Compute psuedorange values and add to dataframe.

    Parameters
    ----------
    gnssRaw : pandas dataframe
        pandas dataframe that holds gnss meassurements
    gnssAnalysis : list
        holds any error messages

    Returns
    -------
    gnssRaw : pandas dataframe
        Dataframe with added columns updated.
    gnssAnalysis : list
        Holds any error messages. This function doesn't actually add any
        error messages, but it is a nice thought.

    Notes
    -----
    Based off of MATLAB code from Google's gps-measurement-tools
    repository: https://github.com/google/gps-measurement-tools. Compare
    with opensource/ProcessGnssMeas.m

    """

    gpsconsts = GPSConsts()
    gnssRaw['Pseudorange_seconds'] = gnssRaw['tRxSeconds'] - gnssRaw['tTxSeconds']
    gnssRaw['Pseudorange_meters'] = gnssRaw['Pseudorange_seconds']*gpsconsts.C
    gnssRaw['Pseudorange_sigma_meters'] = gpsconsts.C * 1e-9 * gnssRaw['ReceivedSvTimeUncertaintyNanos']
    return gnssRaw, gnssAnalysis
