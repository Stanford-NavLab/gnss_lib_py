"""Functions to process broadcast navigation and observations from Rinex.

"""

__authors__ = "Ashwin Kanhere"
__date__ = "19 Jul 2023"

import numpy as np
import georinex as gr

from gnss_lib_py.parsers.navdata import NavData
import gnss_lib_py.utils.constants as consts
from gnss_lib_py.utils.time_conversions import datetime_to_gps_millis


class RinexObs(NavData):
    """Class handling Rinex observation files [1]_.

    The Rinex Observation files (of the format .yyo) contain measured
    pseudoranges, carrier phase, doppler and signal-to-noise ratio
    measurements for multiple constellations and bands.
    This loader converts those file types into a NavData in which
    measurements from different bands are treated as separate measurement
    instances. Inherits from NavData().

    This class has primarily been built with Rinex v3.05 in mind but it
    should also work for prior Rinex versions.


    References
    ----------
    .. [1] https://files.igs.org/pub/data/format/rinex305.pdf


    """

    def __init__(self, input_path):
        """Loading Rinex observation files into a NavData based class.

        Should input path to `.yyo` file.

        Parameters
        ----------
        input_path : string or path-like
            Path to rinex .o file

        """

        obs_file = gr.load(input_path).to_dataframe()
        obs_header = gr.rinexheader(input_path)
        obs_measure_types = obs_header['fields']
        rx_bands = []
        for rx_measures in obs_measure_types.values():
            for single_measure in rx_measures:
                band = single_measure[1]
                if band not in rx_bands:
                    rx_bands.extend(band)

        obs_file.reset_index(inplace=True)
        # Convert time to gps_millis
        gps_millis = [datetime_to_gps_millis(df_row['time']) \
                                for _, df_row in obs_file.iterrows()]
        obs_file['gps_millis'] = gps_millis
        obs_file = obs_file.drop(columns=['time'])
        obs_file = obs_file.rename(columns={"sv":"sv_id"})
        # Convert gnss_sv_id to gnss_id and sv_id (plus gnss_sv_id)
        obs_navdata_raw = NavData(pandas_df=obs_file)
        obs_navdata_raw['gnss_sv_id'] = obs_navdata_raw['sv_id']
        gnss_chars = [sv_id[0] for sv_id in np.atleast_1d(obs_navdata_raw['sv_id'])]
        gnss_nums = [sv_id[1:] for sv_id in np.atleast_1d(obs_navdata_raw['sv_id'])]
        gnss_id = [consts.CONSTELLATION_CHARS[gnss_char] for gnss_char in gnss_chars]
        obs_navdata_raw['gnss_id'] = np.asarray(gnss_id)
        obs_navdata_raw['sv_id'] = np.asarray(gnss_nums, dtype=int)
        # Convert the coded column names to glp standards and extract information
        # into glp row and columns format
        info_rows = ['gps_millis', 'gnss_sv_id', 'sv_id', 'gnss_id']
        super().__init__()
        for band in rx_bands:
            rename_map = {}
            keep_rows = info_rows.copy()
            measure_type_dict = self._measure_type_dict()
            for measure_char, measure_row in measure_type_dict.items():
                measure_band_row = \
                    obs_navdata_raw.find_wildcard_indexes(f'{measure_char}{band}*',
                                                          max_allow=1)
                measure_row_chars = measure_band_row[f'{measure_char}{band}*'][0]
                rename_map[measure_row_chars] = measure_row
                keep_rows.append(measure_row_chars)
            band_navdata = obs_navdata_raw.copy(rows=keep_rows)
            band_navdata.rename(rename_map, inplace=True)
            # Remove the cases with NaNs in the measurements
            for row in rename_map.values():
                band_navdata = band_navdata.where(row, np.nan, 'neq')
            # Assign the gnss_lib_py standard names for signal_type
            rx_constellations = np.unique(band_navdata['gnss_id'])
            signal_type_dict = self._signal_type_dict()
            signal_types = np.empty(len(band_navdata), dtype=object)
            for constellation in rx_constellations:
                signal_type = signal_type_dict[constellation][band]
                signal_types[band_navdata['gnss_id']==constellation] = signal_type
            band_navdata['signal_type'] = signal_types
            if len(self) == 0:
                self.concat(band_navdata, inplace=True)
            else:
                self.concat(band_navdata, inplace=True)
        self.sort('gps_millis', inplace=True)

    @staticmethod
    def _measure_type_dict():
        """Map of Rinex observation measurement types to standard names.

        Returns
        -------
        measure_type_dict : Dict
            Dictionary of the form {rinex_character : measure_name}
        """

        measure_type_dict = {'C': 'raw_pr_m',
                             'L': 'carrier_phase',
                             'D': 'raw_doppler_hz',
                             'S': 'cn0_dbhz'}
        return measure_type_dict

    @staticmethod
    def _signal_type_dict():
        """Dictionary from constellation and signal bands to signal types.

        Returns
        -------
        signal_type_dict : Dict
            Dictionary of the form {constellation_band : {band : signal_type}}
        """
        signal_type_dict = {}
        signal_type_dict['gps'] = {'1' : 'l1',
                                '2' : 'l2',
                                '5' : 'l5'}
        signal_type_dict['glonass'] = {'1' : 'g1',
                                    '4' : 'g1a',
                                    '2' : 'g2',
                                    '6' : 'g2a',
                                    '3' : 'g3'}
        signal_type_dict['galileo'] = {'1' : 'e1',
                                    '5' : 'e5a',
                                    '7' : 'e5b',
                                    '8' : 'e5',
                                    '6' : 'e6'}
        signal_type_dict['sbas'] = {'1' : 'l1',
                                    '5' : 'l5'}
        signal_type_dict['qzss'] = {'1' : 'l1',
                                    '2' : 'l2',
                                    '5' : 'l5',
                                    '6' : 'l6'}
        # beidou needs to be refined because the current level of detail isn't enough
        # to distinguish between different signals
        signal_type_dict['beidou'] = {'2' : 'b1',
                                    '1' : 'b1c',
                                    '5' : 'b2a',
                                    '7' : 'b2b',
                                    '8' : 'b2',
                                    '6' : 'b3'}
        signal_type_dict['irnss'] = {'5' : 'l5',
                                    '9' : 's'}
        return signal_type_dict