"""Functions to process Android measurements.

"""

__authors__ = "Shubh Gupta, Adam Dai, Ashwin Kanhere"
__date__ = "02 Nov 2021"

import os
import sys
import csv
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
# append <path>/gnss_lib_py/gnss_lib_py/ to path

sys.path.append(os.path.dirname(
                os.path.dirname(
                os.path.realpath(__file__))))
from core.constants import GPSConsts
from parsers.measurement import Measurement


class AndroidDerived(Measurement):
    """Class handling derived measurements from Android dataset.

    Inherits from Measurement().
    """
    #NOTE: Inherits __init__() and isn't defined explicitly here because
    # no additional implementations/attributes are needed

    def preprocess(self, input_path):
        """Android specific loading and preprocessing for Measurement()

        Parameters
        ----------
        input_path : string
            Path to measurement csv file

        Returns
        -------
        pd_df : pandas.DataFrame
            Loaded measurements with consistent column names
        """
        pd_df = pd.read_csv(input_path)
        col_map = self._column_map()
        pd_df.rename(columns=col_map, inplace=True)
        return pd_df

    def postprocess(self):
        """Android derived specific postprocessing for Measurement()

        Notes
        -----
        Adds corrected pseudoranges to measurements. Corrections
        implemented from https://www.kaggle.com/carlmcbrideellis/google-smartphone-decimeter-eda
        retrieved on 9th November, 2021
        """
        pr_corrected = self['rawPrM', :] + self['b', :] - \
                    self['isrbM', :] - self['tropoDelayM', :] \
                    - self['ionoDelayM', :]
        self['pseudo'] = pr_corrected
        return None

    @staticmethod
    def _column_map():
        """Map of column names from loaded to gnss_lib_py standard

        Returns
        -------
        col_map : Dict
            Dictionary of the form {old_name : new_name}
        """
        col_map = {'millisSinceGpsEpoch' : 'toe',
                'svid' : 'PRN',
                'xSatPosM' : 'x',
                'ySatPosM' : 'y',
                'zSatPosM' : 'z',
                'xSatVelMps' : 'vx',
                'ySatVelMps' : 'vy',
                'zSatVelMps' : 'vz',
                'satClkBiasM' : 'b',
                'satClkDriftMps' : 'b_dot',
                }
        return col_map


class AndroidRawGnss(Measurement):
    def __init__(self, input_path, verbose=False, use_carrier=False):
        self.verbose = verbose
        self.use_carrier = use_carrier
        self.log = []
        self.time_refs = {'UnixTime' : 0,
                          'UtcTimeNanos' : 0}
        pd_df = self.preprocess(input_path)
        self.pd_df = pd_df
        self.build_measurement(pd_df)
        self.postprocess()
        
    def preprocess(self, input_path):
        """Built on the first parts of make_gnss_dataframe and correct_log
        """
        with open(input_path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0][0] == '#':
                    if 'Raw' in row[0]:
                        read_measures = [row[1:]]
                else:
                    if row[0] == 'Raw':
                        read_measures.append(row[1:])

        measurements = pd.DataFrame(read_measures[1:], columns = read_measures[0], dtype=np.float64)
        measurements = measurements.astype(self._column_types())

        #TODO: Empty columns are being cast to strings. Change to Nan or 0
        single_sv = measurements['Svid'].astype(str).str.len() == 1
        measurements.loc[single_sv, 'Svid'] =\
            '0' + measurements.loc[single_sv, 'Svid'].astype(str)

        # # Compatibility with RINEX files
        measurements.loc[measurements['ConstellationType'].astype(str) == '1', 'Constellation'] = 'G'
        measurements.loc[measurements['ConstellationType'].astype(str) == '3', 'Constellation'] = 'R'
        measurements['SvName'] = measurements['Constellation'].astype(str) + measurements['Svid'].astype(str)

        measurements = measurements.astype(self._column_types())
        # Drop non-GPS measurements
        #TODO: Introduce multi-constellation support
        measurements = measurements.loc[measurements['Constellation'] == 'G']
        #TODO: ^^Explicitly deal with GPS L1 instead of just GPS
        
        col_map = self._column_map()
        measurements.rename(columns=col_map, inplace=True)
        return measurements

    @staticmethod
    def _column_map():
        #TODO: Keep android column names or change?
        col_map = {'SvName' : 'SV',
                    'Svid' : 'PRN'}
        return col_map

    @staticmethod
    def _column_types():
        col_types = {'ConstellationType': int,
                    'Svid': int}
        return col_types

    def postprocess(self):
        """Built on correct_log
        """
        # Check clock and measurement fields
        self._check_gnss_clock()
        self._check_gnss_measurements()
        if self.use_carrier:
            self._check_carrier_phase()
        self._compute_times()
        self._compute_pseudorange()


    def _check_gnss_clock(self):
        """Checks and fixes clock field errors

        Additonal checks added from [1]_.

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
        #TODO: Check if there is way to add a single reference and cite
        # it multiple times
        # list of clock fields
        gnss_clock_fields = [
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
        gnss_fields = self.rows()
        for field in gnss_clock_fields:
            if field not in gnss_fields:
                self._update_log('WARNING: '+field+' (Clock) is missing from GNSS Logger file')
        measure_ok = all(x in gnss_fields for x in ['TimeNanos', 'FullBiasNanos'])
        if not measure_ok:
            self._update_log('FAIL clock check.')
            raise RuntimeError("Measurement file failed clock check")

        # Measurements should be discarded if TimeNanos is empty
        if np.isnan(self["TimeNanos", :]).any():
            self['all'] = self['all', np.logical_not(np.isnan(self["TimeNanos"]))]
            self._update_log('Empty or invalid TimeNanos')

        if 'BiasNanos' not in gnss_fields:
            self["BiasNanos"] = 0
        if 'TimeOffsetNanos' not in gnss_fields:
            self["TimeOffsetNanos"] = 0
        if 'HardwareClockDiscontinuityCount' not in gnss_fields:
            self["HardwareClockDiscontinuityCount"] = 0
            self._update_log('WARNING: Added HardwareClockDiscontinuityCount=0 because it is missing from GNSS Logger file')

        # measurements should be discarded if FullBiasNanos is zero or invalid
        if np.asarray(self["FullBiasNanos", :] >= 0).any():
            self["FullBiasNanos"] = -1*self["FullBiasNanos", :]
            self._update_log('WARNING: FullBiasNanos wrong sign. Should be negative. Auto changing inside check_gnss_clock')

        # Discard measurements if BiasUncertaintyNanos is too large
        # TODO: figure out how to choose this parameter better
        if np.asarray(self["BiasUncertaintyNanos", :] >= 40.).any():
            count = (self["BiasUncertaintyNanos", :] >= 40.).sum()
            self._update_log(str(count) + ' rows with too large BiasUncertaintyNanos')
            self['all'] = self['all', self["BiasUncertaintyNanos", :] < 40.]
            #TODO: Figure out a way that actually works and implement 

        self["allRxMillis"] = (self["TimeNanos", :] - self["FullBiasNanos", :])//1e6


    def _check_gnss_measurements(self):
        """Checks that GNSS measurement fields exist in dataframe.

        Additonal checks added from [1]_.

        Notes
        -----
        Based off of MATLAB code from Google's gps-measurement-tools
        repository: https://github.com/google/gps-measurement-tools. Compare
        with ReportMissingFields() in opensource/ReadGnssLogger.m

        References
        ----------
        .. [1] Fu, Guoyu Michael, Mohammed Khider, and Frank van Diggelen.
        "Android Raw GNSS Measurement Datasets for Precise Positioning."
        Proceedings of the 33rd International Technical Meeting of the
        Satellite Division of The Institute of Navigation (ION GNSS+
        2020). 2020.

        """
        # list of measurement fields
        gnss_measurement_fields = [
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
        gnss_fields = self.rows()
        for field in gnss_measurement_fields:
            if field not in gnss_fields:
                self._update_log('WARNING: '+field+' (Measurement) is missing from GNSS Logger file')

        # Discard measurements if state is neither of following
        STATE_TOW_DECODED = 0x8
        STATE_TOW_KNOWN = 0x4000
        invalid_state_count = np.invert((self["State", :] & STATE_TOW_DECODED).astype(bool) |
                                (self["State", :] & STATE_TOW_KNOWN).astype(bool)).sum()
        if invalid_state_count > 0:
            self._update_log(str(invalid_state_count) + " rows have " + \
                                "state TOW neither decoded nor known")
            valid_state_idx = (self["State", :] & STATE_TOW_DECODED).astype(bool) | \
                            (self["State", :] & STATE_TOW_KNOWN).astype(bool)
            self['all'] = self['all', valid_state_idx]
        # Discard measurements if ReceivedSvTimeUncertaintyNanos is high 
        # TODO: figure out how to choose this parameter better
        if any(self["ReceivedSvTimeUncertaintyNanos", :] >= 150.):
            count = (self["ReceivedSvTimeUncertaintyNanos", :] >= 150.).sum()
            self._update_log(str(count) + ' rows with too large ReceivedSvTimeUncertaintyNanos')
            self['all'] = self['all', self["ReceivedSvTimeUncertaintyNanos", :] < 150.]
            #TODO: Updating all values here

    def _check_carrier_phase(self):
        """Checks that carrier phase measurements are valid

        Checks taken from [1]_.

        References
        ----------
        .. [1] Fu, Guoyu Michael, Mohammed Khider, and Frank van Diggelen.
        "Android Raw GNSS Measurement Datasets for Precise Positioning."
        Proceedings of the 33rd International Technical Meeting of the
        Satellite Division of The Institute of Navigation (ION GNSS+
        2020). 2020.

        """

        # Measurements should be discarded if AdrState violates
        # ADR_STATE_VALID == 1 & ADR_STATE_RESET == 0
        # & ADR_STATE_CYCLE_SLIP == 0
        ADR_STATE_VALID = 0x1
        ADR_STATE_RESET = 0x2
        ADR_STATE_CYCLE_SLIP = 0x4

        invalid_state_count = np.invert((self["AccumulatedDeltaRangeState", :] & ADR_STATE_VALID).astype(bool) &
                            np.invert((self["AccumulatedDeltaRangeState", :] & ADR_STATE_RESET).astype(bool)) &
                            np.invert((self["AccumulatedDeltaRangeState", :] & ADR_STATE_CYCLE_SLIP).astype(bool))).sum()
        if invalid_state_count > 0:
            self._update_log(str(invalid_state_count) + " rows have " + \
                                "ADRstate invalid")
            #TODO: Same operation is happening twice, functionalize or perform once
            valid_states = (self["AccumulatedDeltaRangeState", :] & ADR_STATE_VALID).astype(bool) & \
                            np.invert((self["AccumulatedDeltaRangeState", :] & ADR_STATE_RESET).astype(bool)) & \
                            np.invert((self["AccumulatedDeltaRangeState", :] & ADR_STATE_CYCLE_SLIP).astype(bool))
            self['all'] = self['all' , valid_states]

        # Discard measurements if AccumulatedDeltaRangeUncertaintyMeters
        # is too large 
        #TODO: figure out how to choose this parameter better
        if any(self["AccumulatedDeltaRangeUncertaintyMeters", :] >= 0.15):
            count = (self["AccumulatedDeltaRangeUncertaintyMeters", :] >= 0.15).sum()
            self._update_log(str(count) + 'rows with too large' +
                            ' AccumulatedDeltaRangeUncertaintyMeters')
            self['all'] = self['all', self["AccumulatedDeltaRangeUncertaintyMeters", :] < 0.15]
            #TODO: Updating all values here

    def _compute_times(self):
        """Compute times and epochs for GNSS measurements.

        Additional checks added from [1]_.

        Notes
        -----
        Based off of MATLAB code from Google's gps-measurement-tools
        repository: https://github.com/google/gps-measurement-tools. Compare
        with opensource/ProcessGnssMeas.m

        References
        ----------
        .. [1] Fu, Guoyu Michael, Mohammed Khider, and Frank van Diggelen.
        "Android Raw GNSS Measurement Datasets for Precise Positioning."
        Proceedings of the 33rd International Technical Meeting of the
        Satellite Division of The Institute of Navigation (ION GNSS+
        2020). 2020.

        """
        gpsepoch = datetime(1980, 1, 6, 0, 0, 0)
        WEEKSEC = 604800
        self['gps_week'] = np.floor(-1*self['FullBiasNanos', :]*1e-9/WEEKSEC)
        self['toe_nanos'] = self['TimeNanos', :] - (self['FullBiasNanos', :] + self['BiasNanos', :])

        # Measurements should be discarded if arrival time is negative
        if sum(self['toe_nanos', :] <= 0) > 0:
            self['all'] = self['all', self['toe_nanos'] > 0]
            self._update_log("negative arrival times removed")
        # TODO: Discard measurements if arrival time is too large

        self['tRxNanos'] = (self['TimeNanos', :]+self['TimeOffsetNanos', :])-(self['FullBiasNanos', 0]+self['BiasNanos',:])
        self['tRxSeconds'] = 1e-9*self['tRxNanos', :] - WEEKSEC * self['gps_week', :]
        self['tTxSeconds'] = 1e-9*(self['ReceivedSvTimeNanos', :] + self['TimeOffsetNanos', :])
        leap_not_nan = np.logical_not(np.isnan(self['LeapSecond', :]), dtype=self.arr_dtype)
        self['LeapSecond'] = leap_not_nan *self['LeapSecond', :]
        
        #TODO: Check the timing functions
        self.time_refs['UtcTimeNanoes'] = self['toe_nanos', 0]
        self["UtcTimeNanos"] = self['toe_nanos', :] - self.time_refs['UtcTimeNanos'] - self["LeapSecond", :] * 1E9
        self.time_refs['UnixTime'] = self['toe_nanos', 0]
        self['UnixTime'] = self['toe_nanos', :] - self.time_refs['UtcTimeNanos']
        #TODO: Do we need to store intermediary values and lug them around?

        self['Epoch'] = 0
        epoch_non_zero = self['UnixTime', :] - self['UnixTime', :].shift() > timedelta(milliseconds=200)
        self['Epoch', epoch_non_zero] = 1
        self['Epoch'] = self['Epoch', :].cumsum()
    
    def _compute_pseudorange(self):
        """Compute psuedorange values and add to dataframe.

        Notes
        -----
        Based off of MATLAB code from Google's gps-measurement-tools
        repository: https://github.com/google/gps-measurement-tools. Compare
        with opensource/ProcessGnssMeas.m

        """

        gpsconsts = GPSConsts()
        self['Pseudorange_seconds', :] = self['tRxSeconds', :] - self['tTxSeconds', :]
        self['pseudo'] = self['Pseudorange_seconds', :]*gpsconsts.C
        self['pseudo_sigma'] = gpsconsts.C * 1e-9 * self['ReceivedSvTimeUncertaintyNanos']

    def _update_log(self, msg):
        self.log.append(msg)
        if self.verbose:
            print(msg)

    def _return_times(self, key):
        #TODO: Implement method that uses time reference and returns
        # datetime object
        raise NotImplementedError


class AndroidRawImu(Measurement):
    def __init__(self, input_path, group_time=10):
        self.group_time = group_time
        data_df = self.preprocess(input_path)
        self.build_measurement(data_df)
        self.postprocess()
    
    def preprocess(self, input_path):
        """Read Android raw file and produce IMU dataframe objects

        Parameters
        ----------
        input_path : string
            File location of data file to read.
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

        accel = pd.DataFrame(accel[1:], columns = accel[0], dtype=np.float64)
        gyro = pd.DataFrame(gyro[1:], columns = gyro[0], dtype=np.float64)

        #Drop common columns from gyro and keep values from accel
        gyro.drop(columns=['utcTimeMillis', 'elapsedRealtimeNanos'], inplace=True)
        measurements = pd.concat([accel, gyro], axis=1)
        #NOTE: Assuming pandas index corresponds to measurements order
        measurements.rename(columns=self._column_map(), inplace=True)
        return measurements

    def postprocess(self):
        # Currently not performing any post processing on measurments
        # but need to define method to override abstract method
        pass

    @staticmethod
    def _column_map():
        col_map = {'AccelXMps2' : 'acc_x',
                'AccelYMps2' : 'acc_y',
                'AccelZMps2' : 'acc_z',
                'GyroXRadPerSec' : 'omega_x',
                'GyroYRadPerSec' : 'omega_y',
                'GyroZRadPerSec' : 'omega_z',
                }
        return col_map


class AndroidRawFixes(Measurement):
    def preprocess(self, input_path):
        with open(input_path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0][0] == '#':
                    if 'Fix' in row[0]:
                        android_fixes = [row[1:]]
                else:
                    if row[0] == 'Fix':
                        android_fixes.append(row[1:])

        fix_df = pd.DataFrame(android_fixes[1:], columns = android_fixes[0])
        return fix_df

    def postprocess(self):
        # Currently not performing any post processing on measurments
        # but need to define to override abstract method
        pass


def make_csv(input_path, field):
    #TODO: Add handling for first n times field appear
    """Write specific data types from a GNSS android log to a CSV.

    Parameters
    ----------
    input_path : string
        File location of data file to read.
    fields : list of strings
        Type of data to extract. Valid options are either "Raw",
        "Accel", "Gyro", "Mag", or "Fix".

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