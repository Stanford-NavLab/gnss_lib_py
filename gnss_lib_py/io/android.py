"""Functions to process Android measurements.

"""

__authors__ = "Shubh Gupta, Adam Dai, Ashwin Kanhere"
__date__ = "02 Nov 2021"

import os
import sys
import csv
from datetime import datetime, timedelta
# append <path>/gnss_lib_py/gnss_lib_py/ to path
sys.path.append(os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__))))

import numpy as np
import pandas as pd

from core.constants import GPSConsts


class AndroidDerived():
    def __init__(self, input_path):
        derived_df = pd.read_csv(input_path)
        num_times = len(derived_df)
        # num_values = len(derived_df.columns)
        self.arr_dtype = np.float64
        self.array = np.empty([0, num_times], dtype= self.arr_dtype) 
        # Defining using an empty array to conserve space and not maintain huge duplicates
        self.map = {col_name: idx for idx, col_name in enumerate(derived_df.columns)}
        str_bool = {col_name: type(derived_df.loc[0, col_name])==str for col_name in self.map.keys()}
        # Assuming that the data type of all objects in a single series is the same
        self.str_map = {}
        for key in self.map.keys():
            # indexing by key to maintain same order as eventual map
            val = str_bool[key]
            # print(key)
            values = derived_df.loc[:, key]
            new_values = np.copy(values)
            if val:
                string_vals = np.unique(derived_df.loc[:, key])
                val_dict = {idx : string_name for idx, string_name in enumerate(string_vals)}
                self.str_map[key] = val_dict

                for key, val in val_dict.items():
                    new_values[values==val] = key
                # Copy set to false to prevent memory overflows
                new_values = new_values.astype(self.arr_dtype, copy=False)
            else:
                self.str_map[key] = {}
            new_values = np.reshape(new_values, [1, -1])
            # print('self.array shape: ', np.shape(self.array))
            # print('new_values shape: ', np.shape(new_values))
            self.array = np.vstack((self.array, new_values))

    def get_strings(self, key):
        values_int = self.array[self.map[key],:]
        values_str = values_int.astype(str, copy=True) 
        # True by default but making explicit for clarity
        for key, val in self.str_map[key].items():
            values_str[values_int==key] = val
        return values_str


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

def check_gnss_clock(gnss_raw, gnssAnalysis):
    """Checks and fixes clock field errors

    Additonal checks added from [1]_.

    Parameters
    ----------
    gnss_raw : pandas dataframe
        pandas dataframe that holds gnss meassurements
    gnssAnalysis : list
        holds any error messages

    Returns
    -------
    gnss_raw : pandas dataframe
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
        if field not in gnss_raw.head():
            gnssAnalysis.append('WARNING: '+field+' (Clock) is missing from GNSS Logger file')
        else:
            gnss_raw.loc[:,field] = pd.to_numeric(gnss_raw[field])
    ok = all(x in gnss_raw.head() for x in ['TimeNanos', 'FullBiasNanos'])
    if not ok:
        gnssAnalysis.append('FAIL Clock check')
        return gnss_raw, gnssAnalysis

    # Measurements should be discarded if TimeNanos is empty
    if gnss_raw["TimeNanos"].isnull().values.any():
        gnss_raw.dropna(how = "any", subset = ["TimeNanos"],
                       inplace = True)
        gnssAnalysis.append('empty or invalid TimeNanos')

    if 'BiasNanos' not in gnss_raw.head():
        gnss_raw.loc[:,'BiasNanos'] = 0
    if 'TimeOffsetNanos' not in gnss_raw.head():
        gnss_raw.loc[:,'TimeOffsetNanos'] = 0
    if 'HardwareClockDiscontinuityCount' not in gnss_raw.head():
        gnss_raw.loc[:,'HardwareClockDiscontinuityCount'] = 0
        gnssAnalysis.append('WARNING: Added HardwareClockDiscontinuityCount=0 because it is missing from GNSS Logger file')

    # measurements should be discarded if FullBiasNanos is zero or invalid
    if any(gnss_raw.FullBiasNanos >= 0):
        gnss_raw.FullBiasNanos = -1*gnss_raw.FullBiasNanos
        gnssAnalysis.append('WARNING: FullBiasNanos wrong sign. Should be negative. Auto changing inside check_gnss_clock')

    # Measurements should be discarded if BiasUncertaintyNanos is too
    # large ## TODO: figure out how to choose this parameter better
    if any(gnss_raw.BiasUncertaintyNanos >= 40.):
        count = (gnss_raw["BiasUncertaintyNanos"] >= 40.).sum()
        gnssAnalysis.append(str(count) +
         ' rows with too large BiasUncertaintyNanos')
        gnss_raw = gnss_raw[gnss_raw["BiasUncertaintyNanos"] < 40.]


    gnss_raw = gnss_raw.assign(allRxMillis = ((gnss_raw.TimeNanos - gnss_raw.FullBiasNanos)/1e6))
    # gnss_raw['allRxMillis'] = ((gnss_raw.TimeNanos - gnss_raw.FullBiasNanos)/1e6)
    return gnss_raw, gnssAnalysis


def check_gnss_measurements(gnss_raw, gnssAnalysis):
    """Checks that GNSS measurement fields exist in dataframe.

    Additonal checks added from [2]_.

    Parameters
    ----------
    gnss_raw : pandas dataframe
        pandas dataframe that holds gnss meassurements
    gnssAnalysis : list
        holds any error messages

    Returns
    -------
    gnss_raw : pandas dataframe
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
        if field not in gnss_raw.head():
            gnssAnalysis.append('WARNING: '+field+' (Measurement) is missing from GNSS Logger file')

    # measurements should be discarded if state is neither
    # STATE_TOW_DECODED nor STATE_TOW_KNOWN
    gnss_raw = gnss_raw.assign(State=pd.to_numeric(gnss_raw['State']))
    STATE_TOW_DECODED = 0x8
    STATE_TOW_KNOWN = 0x4000
    invalid_state_count = np.invert((gnss_raw["State"] & STATE_TOW_DECODED).astype(bool) |
                              (gnss_raw["State"] & STATE_TOW_KNOWN).astype(bool)).sum()
    if invalid_state_count > 0:
        gnssAnalysis.append(str(invalid_state_count) + " rows have " + \
                            "state TOW neither decoded nor known")
        gnss_raw = gnss_raw[(gnss_raw["State"] & STATE_TOW_DECODED).astype(bool) |
                          (gnss_raw["State"] & STATE_TOW_KNOWN).astype(bool)]

    # Measurements should be discarded if ReceivedSvTimeUncertaintyNanos
    # is high ## TODO: figure out how to choose this parameter better
    if any(gnss_raw.ReceivedSvTimeUncertaintyNanos >= 150.):
        count = (gnss_raw["ReceivedSvTimeUncertaintyNanos"] >= 150.).sum()
        gnssAnalysis.append(str(count) +
         ' rows with too large ReceivedSvTimeUncertaintyNanos')
        gnss_raw = gnss_raw[gnss_raw["ReceivedSvTimeUncertaintyNanos"] < 150.]

    # convert multipath indicator to numeric
    gnss_raw = gnss_raw.assign(MultipathIndicator=pd.to_numeric(gnss_raw['MultipathIndicator']))

    return gnss_raw, gnssAnalysis

def check_carrier_phase(gnss_raw, gnssAnalysis):
    """Checks that carrier phase measurements.

    Checks taken from [3]_.

    Parameters
    ----------
    gnss_raw : pandas dataframe
        pandas dataframe that holds gnss meassurements
    gnssAnalysis : list
        holds any error messages

    Returns
    -------
    gnss_raw : pandas dataframe
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
    gnss_raw = gnss_raw.assign(AccumulatedDeltaRangeState=pd.to_numeric(gnss_raw['AccumulatedDeltaRangeState']))
    ADR_STATE_VALID = 0x1
    ADR_STATE_RESET = 0x2
    ADR_STATE_CYCLE_SLIP = 0x4

    invalid_state_count = np.invert((gnss_raw["AccumulatedDeltaRangeState"] & ADR_STATE_VALID).astype(bool) &
                          np.invert((gnss_raw["AccumulatedDeltaRangeState"] & ADR_STATE_RESET).astype(bool)) &
                          np.invert((gnss_raw["AccumulatedDeltaRangeState"] & ADR_STATE_CYCLE_SLIP).astype(bool))).sum()
    if invalid_state_count > 0:
        gnssAnalysis.append(str(invalid_state_count) + " rows have " + \
                            "ADRstate invalid")
        gnss_raw = gnss_raw[(gnss_raw["AccumulatedDeltaRangeState"] & ADR_STATE_VALID).astype(bool) &
                          np.invert((gnss_raw["AccumulatedDeltaRangeState"] & ADR_STATE_RESET).astype(bool)) &
                          np.invert((gnss_raw["AccumulatedDeltaRangeState"] & ADR_STATE_CYCLE_SLIP).astype(bool))]

    # Measurements should be discarded if AccumulatedDeltaRangeUncertaintyMeters
    # is too large ## TODO: figure out how to choose this parameter better
    gnss_raw = gnss_raw.assign(AccumulatedDeltaRangeUncertaintyMeters=pd.to_numeric(gnss_raw['AccumulatedDeltaRangeUncertaintyMeters']))
    if any(gnss_raw.AccumulatedDeltaRangeUncertaintyMeters >= 0.15):
        count = (gnss_raw["AccumulatedDeltaRangeUncertaintyMeters"] >= 0.15).sum()
        gnssAnalysis.append(str(count) +
         ' rows with too large AccumulatedDeltaRangeUncertaintyMeters')
        gnss_raw = gnss_raw[gnss_raw["AccumulatedDeltaRangeUncertaintyMeters"] < 0.15]

    return gnss_raw, gnssAnalysis


def compute_times(gnss_raw, gnssAnalysis):
    """Compute times and epochs for GNSS measurements.

    Additional checks added from [4]_.

    Parameters
    ----------
    gnss_raw : pandas dataframe
        pandas dataframe that holds gnss meassurements
    gnssAnalysis : list
        holds any error messages

    Returns
    -------
    gnss_raw : pandas dataframe
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
    gnss_raw['GpsWeekNumber'] = np.floor(-1*gnss_raw['FullBiasNanos']*1e-9/WEEKSEC)
    gnss_raw['GpsTimeNanos'] = gnss_raw['TimeNanos'] - (gnss_raw['FullBiasNanos'] + gnss_raw['BiasNanos'])

    # Measurements should be discarded if arrival time is negative
    if sum(gnss_raw['GpsTimeNanos'] <= 0) > 0:
        gnss_raw = gnss_raw[gnss_raw['GpsTimeNanos'] > 0]
        gnssAnalysis.append("negative arrival times removed")
    # TODO: Measurements should be discarded if arrival time is
    # unrealistically large

    gnss_raw['tRxNanos'] = (gnss_raw['TimeNanos']+gnss_raw['TimeOffsetNanos'])-(gnss_raw['FullBiasNanos'].iloc[0]+gnss_raw['BiasNanos'].iloc[0])
    gnss_raw['tRxSeconds'] = 1e-9*gnss_raw['tRxNanos'] - WEEKSEC * gnss_raw['GpsWeekNumber']
    gnss_raw['tTxSeconds'] = 1e-9*(gnss_raw['ReceivedSvTimeNanos'] + gnss_raw['TimeOffsetNanos'])
    gnss_raw['LeapSecond'] = gnss_raw['LeapSecond'].fillna(0)
    gnss_raw["UtcTimeNanos"] = pd.to_datetime(gnss_raw['GpsTimeNanos']  - gnss_raw["LeapSecond"] * 1E9, utc = True, origin=gpsepoch)
    gnss_raw['UnixTime'] = pd.to_datetime(gnss_raw['GpsTimeNanos'], utc = True, origin=gpsepoch)
    # TODO: Check if UnixTime is the same as UtcTime, if so remove it

    gnss_raw['Epoch'] = 0
    gnss_raw.loc[gnss_raw['UnixTime'] - gnss_raw['UnixTime'].shift() > timedelta(milliseconds=200), 'Epoch'] = 1
    gnss_raw['Epoch'] = gnss_raw['Epoch'].cumsum()
    return gnss_raw, gnssAnalysis

def compute_pseudorange(gnss_raw, gnssAnalysis):
    """Compute psuedorange values and add to dataframe.

    Parameters
    ----------
    gnss_raw : pandas dataframe
        pandas dataframe that holds gnss meassurements
    gnssAnalysis : list
        holds any error messages

    Returns
    -------
    gnss_raw : pandas dataframe
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
    gnss_raw['Pseudorange_seconds'] = gnss_raw['tRxSeconds'] - gnss_raw['tTxSeconds']
    gnss_raw['Pseudorange_meters'] = gnss_raw['Pseudorange_seconds']*gpsconsts.C
    gnss_raw['Pseudorange_sigma_meters'] = gpsconsts.C * 1e-9 * gnss_raw['ReceivedSvTimeUncertaintyNanos']
    return gnss_raw, gnssAnalysis
