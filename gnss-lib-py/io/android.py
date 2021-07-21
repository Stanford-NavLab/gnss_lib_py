########################################################################
# Author(s):    Shubh Gupta, Adam Dai
# Date:         13 July 2021
# Desc:         Functions to process Android measurements
########################################################################

import numpy as np
import csv
import pandas as pd
from datetime import datetime, timedelta
from utils.constants import GPSConsts


# Extract data different timesteps
def extract_timedata(input_path):
    """Shubh wrote these (may have google MATLAB counterpart)
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

#Extract data continuous and make a csv
# field - 'Raw', 'Accel', 'Gyro'
def make_csv(input_path, field):
    """Shubh wrote these (may have google MATLAB counterpart)
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

# Read Android raw file and produce gnss dataframe objects
def make_gnss_dataframe(input_path, verbose=False):
    """Shubh wrote this
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

    return fix_log(measurements, verbose=verbose), android_fixes

# Read Android raw file and produce imu dataframe objects
def make_imu_dataframe(input_path, verbose=False):
    """Adam Dai wrote this
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

# Compute required quantities from the log and check for errors
def fix_log(measurements, verbose=False):
    """Shubh wrote this
    """
    # Add leading 0
    measurements.loc[measurements['Svid'].str.len() == 1, 'Svid'] = '0' + measurements['Svid']

    # Compatibility with RINEX files
    measurements.loc[measurements['ConstellationType'] == '1', 'Constellation'] = 'G'
    measurements.loc[measurements['ConstellationType'] == '3', 'Constellation'] = 'R'
    measurements['SvName'] = measurements['Constellation'] + measurements['Svid']

    # Drop non-GPS measurements
    measurements = measurements.loc[measurements['Constellation'] == 'G']

    # Convert columns to numeric representation
    measurements['Cn0DbHz'] = pd.to_numeric(measurements['Cn0DbHz'])
    measurements['TimeNanos'] = pd.to_numeric(measurements['TimeNanos'])
    measurements['FullBiasNanos'] = pd.to_numeric(measurements['FullBiasNanos'])
    measurements['ReceivedSvTimeNanos']  = pd.to_numeric(measurements['ReceivedSvTimeNanos'])
    measurements['PseudorangeRateMetersPerSecond'] = pd.to_numeric(measurements['PseudorangeRateMetersPerSecond'])
    measurements['ReceivedSvTimeUncertaintyNanos'] = pd.to_numeric(measurements['ReceivedSvTimeUncertaintyNanos'])

    # Check clock fields
    error_logs = []
    measurements, error_logs = check_gnss_clock(measurements, error_logs)
    measurements, error_logs = check_gnss_measurements(measurements, error_logs)
    measurements, error_logs = compute_times(measurements, error_logs)
    measurements, error_logs = compute_pseudorange(measurements, error_logs)

    if verbose:
        if len(error_logs)>0:
            print("Following problems detected:")
            print(error_logs)
        else:
            print("No problems detected.")
    return measurements

#Check and fix clock field errors
def check_gnss_clock(gnssRaw, gnssAnalysis):
    """Shubh wrote this
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
            gnssRaw[field] = pd.to_numeric(gnssRaw[field])
    ok = all(x in gnssRaw.head() for x in ['TimeNanos', 'FullBiasNanos'])
    if 'BiasNanos' not in gnssRaw.head():
        gnssRaw['BiasNanos'] = 0
    if 'TimeOffsetNanos' not in gnssRaw.head():
        gnssRaw['TimeOffsetNanos'] = 0
    if 'HardwareClockDiscontinuityCount' not in gnssRaw.head():
        gnssRaw['HardwareClockDiscontinuityCount'] = 0
        gnssAnalysis.append('WARNING: Added HardwareClockDiscontinuityCount=0 because it is missing from GNSS Logger file')
    if any(gnssRaw.FullBiasNanos > 0):
        gnssRaw.FullBiasNanos = -1*gnssRaw.FullBiasNanos
        gnssAnalysis.append('WARNING: FullBiasNanos wrong sign. Should be negative. Auto changing inside check_gnss_clock')
    if not ok:
        gnssAnalysis.append('FAIL Clock check')
    gnssRaw['allRxMillis'] = ((gnssRaw.TimeNanos - gnssRaw.FullBiasNanos)/1e6)
    return gnssRaw, gnssAnalysis

#Check GNSS Measurements
def check_gnss_measurements(gnssRaw, gnssAnalysis):
    """Shubh wrote this (has GOOGLE MATLAB counterpart)
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
    return gnssRaw, gnssAnalysis

# Compute times and epochs
def compute_times(gnssRaw, gnssAnalysis):
    """Shubh wrote this (has GOOGLE MATLAB counterpart)
    """
    gpsepoch = datetime(1980, 1, 6, 0, 0, 0)
    gnssRaw['GpsWeekNumber'] = np.floor(-1*gnssRaw['FullBiasNanos']*1e-9/WEEKSEC)
    gnssRaw['GpsTimeNanos'] = gnssRaw['TimeNanos'] - (gnssRaw['FullBiasNanos'] - gnssRaw['BiasNanos'])
    gnssRaw['tRxNanos'] = (gnssRaw['TimeNanos']+gnssRaw['TimeOffsetNanos'])-(gnssRaw['FullBiasNanos'].iloc[0]+gnssRaw['BiasNanos'].iloc[0])
    gnssRaw['tRxSeconds'] = 1e-9*gnssRaw['tRxNanos'] - WEEKSEC * gnssRaw['GpsWeekNumber']
    gnssRaw['tTxSeconds'] = 1e-9*(gnssRaw['ReceivedSvTimeNanos'] + gnssRaw['TimeOffsetNanos'])
    gnssRaw['UnixTime'] = pd.to_datetime(gnssRaw['GpsTimeNanos'], utc = True, origin=gpsepoch)

    gnssRaw['Epoch'] = 0
    gnssRaw.loc[gnssRaw['UnixTime'] - gnssRaw['UnixTime'].shift() > timedelta(milliseconds=200), 'Epoch'] = 1
    gnssRaw['Epoch'] = gnssRaw['Epoch'].cumsum()
    return gnssRaw, gnssAnalysis

def compute_pseudorange(gnssRaw, gnssAnalysis):
    """Shubh wrote this (has GOOGLE MATLAB counterpart)
    """
    gpsconsts = GPSConsts()
    gnssRaw['Pseudorange_seconds'] = gnssRaw['tRxSeconds'] - gnssRaw['tTxSeconds']
    gnssRaw['Pseudorange_meters'] = gnssRaw['Pseudorange_seconds']*gpsconsts.C
    gnssRaw['Pseudorange_sigma_meters'] = gpsconsts.C * 1e-9 * gnssRaw['ReceivedSvTimeUncertaintyNanos']
    return gnssRaw, gnssAnalysis