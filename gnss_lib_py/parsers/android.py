"""Functions to process Android measurements.

"""

__authors__ = "Shubh Gupta, Adam Dai, Ashwin Kanhere"
__date__ = "02 Nov 2021"

import os
import csv

import numpy as np
import pandas as pd

from gnss_lib_py.parsers.navdata import NavData


class AndroidDerived(NavData):
    """Class handling derived measurements from Android dataset.

    Inherits from NavData().
    """
    def __init__(self, input_path):
        """Android specific loading and preprocessing

        Parameters
        ----------
        input_path : string
            Path to measurement csv file

        Returns
        -------
        pd_df : pd.DataFrame
            Loaded measurements with consistent column names
        """

        super().__init__(csv_path=input_path)

    def postprocess(self):
        """Android derived specific postprocessing

        Notes
        -----
        Adds corrected pseudoranges to measurements. Corrections
        implemented from https://www.kaggle.com/c/google-smartphone-decimeter-challenge/data
        retrieved on 12 January, 2022
        """
        pr_corrected = self['raw_pr_m', :] \
                     + self['b_sv_m', :] \
                     - self['intersignal_bias_m', :] \
                     - self['tropo_delay_m', :] \
                     - self['iono_delay_m', :]
        self['corr_pr_m'] = pr_corrected

    @staticmethod
    def _row_map():
        """Map of column names from loaded to gnss_lib_py standard

        Returns
        -------
        row_map : Dict
            Dictionary of the form {old_name : new_name}
        """
        row_map = {'collectionName' : 'trace_name',
                   'phoneName' : 'rx_name',
                   'constellationType' : 'gnss_id',
                   'svid' : 'sv_id',
                   'signalType' : 'signal_type',
                   'xSatPosM' : 'x_sv_m',
                   'ySatPosM' : 'y_sv_m',
                   'zSatPosM' : 'z_sv_m',
                   'xSatVelMps' : 'vx_sv_mps',
                   'ySatVelMps' : 'vy_sv_mps',
                   'zSatVelMps' : 'vz_sv_mps',
                   'satClkBiasM' : 'b_sv_m',
                   'satClkDriftMps' : 'b_dot_sv_mps',
                   'rawPrM' : 'raw_pr_m',
                   'rawPrUncM' : 'raw_pr_sigma_m',
                   'isrbM' : 'intersignal_bias_m',
                   'ionoDelayM' : 'iono_delay_m',
                   'tropoDelayM' : 'tropo_delay_m',
                   }
        return row_map


class AndroidRawImu(NavData):
    """Class handling IMU measurements from raw Android dataset.

    Inherits from NavData().
    """
    def __init__(self, input_path, group_time=10):
        self.group_time = group_time
        pd_df = self.preprocess(input_path)
        super().__init__(pandas_df=pd_df)


    def preprocess(self, input_path):
        """Read Android raw file and produce IMU dataframe objects

        Parameters
        ----------
        input_path : string
            File location of data file to read.

        Returns
        -------
        accel : pd.DataFrame
            Dataframe that contains the accel measurements from the log.
        gyro : pd.DataFrame
            Dataframe that contains the gyro measurements from the log.

        """
        with open(input_path, encoding="utf8") as csvfile:
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
        #NOTE: Override times of gyro measurements with corresponding
        # accel times
        return measurements

    @staticmethod
    def _row_map():
        row_map = {'AccelXMps2' : 'acc_x_mps2',
                   'AccelYMps2' : 'acc_y_mps2',
                   'AccelZMps2' : 'acc_z_mps2',
                   'GyroXRadPerSec' : 'ang_vel_x_radps',
                   'GyroYRadPerSec' : 'ang_vel_y_radps',
                   'GyroZRadPerSec' : 'ang_vel_z_radps',
                   }
        return row_map


class AndroidRawFixes(NavData):
    """Class handling location fix measurements from raw Android dataset.

    Inherits from NavData().
    """
    def __init__(self, input_path):
        pd_df = self.preprocess(input_path)
        super().__init__(pandas_df=pd_df)


    def preprocess(self, input_path):
        """Read Android raw file and produce location fix dataframe objects

        Parameters
        ----------
        input_path : string
            File location of data file to read.

        Returns
        -------
        fix_df : pd.DataFrame
            Dataframe that contains the location fixes from the log.

        """
        with open(input_path, encoding="utf8") as csvfile:
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


def make_csv(input_path, output_directory, field, show_path=False):
    """Write specific data types from a GNSS android log to a CSV.

    Parameters
    ----------
    input_path : string
        File location of data file to read.
    output_directory : string
        Directory where new csv file should be created
    fields : list of strings
        Type of data to extract. Valid options are either "Raw",
        "Accel", "Gyro", "Mag", or "Fix".

    Returns
    -------
    output_path : string
        New file location of the exported CSV.

    Notes
    -----
    Based off of MATLAB code from Google's gps-measurement-tools
    repository: https://github.com/google/gps-measurement-tools. Compare
    with MakeCsv() in opensource/ReadGnssLogger.m

    """
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    output_path = os.path.join(output_directory, field + ".csv")
    with open(output_path, 'w', encoding="utf8") as out_csv:
        writer = csv.writer(out_csv)
        with open(input_path, 'r', encoding="utf8") as in_txt:
            for line in in_txt:
                # Comments in the log file
                if line[0] == '#':
                    # Remove initial '#', spaces, trailing newline
                    # and split using commas as delimiter
                    line_data = line[2:].rstrip('\n').replace(" ","").split(",")
                    if line_data[0] == field:
                        writer.writerow(line_data[1:])
                # Data in file
                else:
                    # Remove spaces, trailing newline and split using commas as delimiter
                    line_data = line.rstrip('\n').replace(" ","").split(",")
                    if line_data[0] == field:
                        writer.writerow(line_data[1:])
    if show_path:
        print(output_path)

    return output_path


class AndroidRawGnss(NavData):
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
        gnss_fields = self.rows
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
        gnss_fields = self.rows
        for field in gnss_measurement_fields:
            if field not in gnss_fields:
                self._update_log('WARNING: '+field+' (Measurement) is missing from GNSS Logger file')

        # Discard measurements if state is neither of following
        STATE_TOW_DECODED = 0x8
        STATE_TOW_KNOWN = 0x4000
        decoded_state = np.logical_and(self['State', :], STATE_TOW_DECODED)
        known_state = np.logical_and(self["State", :], STATE_TOW_KNOWN)
        valid_state_idx = decoded_state
        invalid_state_count = np.invert(np.logical_or(decoded_state, known_state)).sum()
        if invalid_state_count > 0:
            self._update_log(str(invalid_state_count) + " rows have " + \
                                "state TOW neither decoded nor known")
            valid_state_idx = (self["State", :] & STATE_TOW_DECODED).astype(bool) | \
                            (self["State", :] & STATE_TOW_KNOWN).astype(bool)
            self['all'] = self['all', valid_state_idx]
        # Discard measurements if ReceivedSvTimeUncertaintyNanos is high
        # TODO: figure out how to choose this parameter better
        if np.any(self["ReceivedSvTimeUncertaintyNanos", :] >= 150.):
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
        if np.any(self['toe_nanos', :] <= 0) > 0:
            self['all'] = self['all', self['toe_nanos'] > 0]
            self._update_log("negative arrival times removed")
        # TODO: Discard measurements if arrival time is too large

        self['tRxNanos'] = (self['TimeNanos', :]+self['TimeOffsetNanos', :])-(self['FullBiasNanos', 0]+self['BiasNanos',:])
        self['tRxSeconds'] = 1e-9*self['tRxNanos', :] - WEEKSEC * self['gps_week', :]
        self['tTxSeconds'] = 1e-9*(self['ReceivedSvTimeNanos', :] + self['TimeOffsetNanos', :])
        leap_nan = np.isnan(self['LeapSecond', :])
        leap_not_nan = np.logical_not(leap_nan)
        # leap_not_nan = np.logical_not(np.isnan(self['LeapSecond', :]), dtype=self.arr_dtype)
        self['LeapSecond'] = leap_not_nan *self['LeapSecond', :]
        #TODO: Check the timing functions
        self.time_refs['UtcTimeNanoes'] = self['toe_nanos', 0]
        self["UtcTimeNanos"] = self['toe_nanos', :] - self.time_refs['UtcTimeNanos'] - self["LeapSecond", :] * 1E9
        self.time_refs['UnixTime'] = self['toe_nanos', 0]
        self['UnixTime'] = self['toe_nanos', :] - self.time_refs['UtcTimeNanos']
        #TODO: Do we need to store intermediary values and lug them around?

        self['Epoch'] = 0*np.zeros([len(self), 1])
        print(type(self['UnixTime', 0][0]))
        print(self['UnixTime', :]- np.roll(self['UnixTime', :], -1))
        # epoch_non_zero = np.where(self['UnixTime', :] - np.roll(self['UnixTime', :], 1) > timedelta(milliseconds=200))
        #TODO: Figure out a possibly better way of doing this
        # epoch_non_zero = np.where(self['UnixTime', :] - np.roll(self['UnixTime', :], 1) > 0.2)
        # self['Epoch', epoch_non_zero] = 1
        # print(np.cumsum(self['Epoch', :]))
        #TODO: Fix the method for calculating self['Epoch', :]
        # self['Epoch'] = np.cumsum(self['Epoch', :])

    def _compute_pseudorange(self):
        """Compute psuedorange values and add to dataframe.
        Notes
        -----
        Based off of MATLAB code from Google's gps-measurement-tools
        repository: https://github.com/google/gps-measurement-tools. Compare
        with opensource/ProcessGnssMeas.m
        """
        self['Pseudorange_seconds'] = self['tRxSeconds', :] - self['tTxSeconds', :]
        print(self['Pseudorange_seconds', :]*gpsconsts.C)
        self['pseudo'] = self['Pseudorange_seconds', :]*gpsconsts.C
        self['pseudo_sigma'] = gpsconsts.C * 1e-9 * self['ReceivedSvTimeUncertaintyNanos', :]

    def _update_log(self, msg):
        self.log.append(msg)
        if self.verbose:
            print(msg)

    def _return_times(self, key):
        #TODO: Implement method that uses time reference and returns
        # datetime object
        raise NotImplementedError

