"""Parses Rinex .o files"""

__authors__ = "Ashwin Kanhere"
__date__ = "26 July 2023"

from datetime import timezone

import numpy as np
import georinex as gr

from gnss_lib_py.navdata.navdata import NavData
import gnss_lib_py.utils.constants as consts
from gnss_lib_py.utils.time_conversions import datetime_to_gps_millis
from gnss_lib_py.navdata.operations import sort, concat

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
                band = single_measure[1:]
                if band not in rx_bands:
                    rx_bands.append(band)
        obs_file.dropna(how='all', inplace=True)
        obs_file.reset_index(inplace=True)
        # Convert time to gps_millis
        datetime_series = [d.to_pydatetime().replace(tzinfo=timezone.utc)
                           for d in obs_file["time"]]
        obs_file['gps_millis'] = datetime_to_gps_millis(datetime_series)
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
                measure_band_row = measure_char + band
                rename_map[measure_band_row] = measure_row
                keep_rows.append(measure_char + band)
                if measure_band_row not in obs_navdata_raw.rows:
                    obs_navdata_raw[measure_band_row] = np.array(len(obs_navdata_raw)*[np.nan])
            band_navdata = obs_navdata_raw.copy(rows=keep_rows)
            band_navdata.rename(rename_map, inplace=True)
            # Remove the cases with NaNs in the measurements
            nan_indexes = np.argwhere(np.isnan(band_navdata[["carrier_phase",
            	             "raw_doppler_hz",
                             "cn0_dbhz"]]).all(axis=0))[:,0].tolist()
            # Remove the cases with NaNs in the pseudorange
            nan_indexes += np.argwhere(np.isnan(band_navdata[["raw_pr_m"
                                                             ]]))[:,0].tolist()
            nan_indexes = sorted(list(set(nan_indexes)))
            if len(nan_indexes) > 0:
                band_navdata.remove(cols=nan_indexes,inplace=True)

            # Assign the gnss_lib_py standard names for signal_type
            rx_constellations = np.unique(band_navdata['gnss_id'])
            signal_type_dict = self._signal_type_dict()
            signal_types = np.empty(len(band_navdata), dtype=object)
            observation_codes = np.empty(len(band_navdata), dtype=object)
            for constellation in rx_constellations:
                signal_type = signal_type_dict[constellation][band]
                signal_types[band_navdata['gnss_id']==constellation] = signal_type
                observation_codes[band_navdata['gnss_id']==constellation] = band
            band_navdata['signal_type'] = signal_types
            band_navdata['observation_code'] = observation_codes
            if len(self) == 0:
                concat_navdata = concat(self, band_navdata)
            else:
                concat_navdata = concat(self, band_navdata)

            self.array = concat_navdata.array
            self.map = concat_navdata.map
            self.str_map = concat_navdata.str_map
            self.orig_dtypes = concat_navdata.orig_dtypes.copy()
            
        sort(self,'gps_millis', inplace=True)

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

        Transformations from Section 5.1 in [2]_ and 5.2.17 from [3]_.

        Returns
        -------
        signal_type_dict : Dict
            Dictionary of the form {constellation_band : {band : signal_type}}

        References
        ----------
        .. [2] https://files.igs.org/pub/data/format/rinex304.pdf
        .. [3] https://files.igs.org/pub/data/format/rinex305.pdf

        """
        signal_type_dict = {}
        signal_type_dict['gps'] = {'1C' : 'l1',
                                   '1S' : 'l1',
                                   '1L' : 'l1',
                                   '1X' : 'l1',
                                   '1P' : 'l1',
                                   '1W' : 'l1',
                                   '1Y' : 'l1',
                                   '1M' : 'l1',
                                   '1N' : 'l1',
                                   '2C' : 'l2',
                                   '2D' : 'l2',
                                   '2S' : 'l2',
                                   '2L' : 'l2',
                                   '2X' : 'l2',
                                   '2P' : 'l2',
                                   '2W' : 'l2',
                                   '2Y' : 'l2',
                                   '2M' : 'l2',
                                   '2N' : 'l2',
                                   '5I' : 'l5',
                                   '5Q' : 'l5',
                                   '5X' : 'l5',
                                   }
        signal_type_dict['glonass'] = {'1C' : 'g1',
                                       '1P' : 'g1',
                                       '4A' : 'g1a',
                                       '4B' : 'g1a',
                                       '4X' : 'g1a',
                                       '2C' : 'g2',
                                       '2P' : 'g2',
                                       '6A' : 'g2a',
                                       '6B' : 'g2a',
                                       '6X' : 'g2a',
                                       '3I' : 'g3',
                                       '3Q' : 'g3',
                                       '3X' : 'g3',
                                       }
        signal_type_dict['galileo'] = {'1A' : 'e1',
                                       '1B' : 'e1',
                                       '1C' : 'e1',
                                       '1X' : 'e1',
                                       '1Z' : 'e1',
                                       '5I' : 'e5a',
                                       '5Q' : 'e5a',
                                       '5X' : 'e5a',
                                       '7I' : 'e5b',
                                       '7Q' : 'e5b',
                                       '7X' : 'e5b',
                                       '8I' : 'e5',
                                       '8Q' : 'e5',
                                       '8X' : 'e5',
                                       '6A' : 'e6',
                                       '6B' : 'e6',
                                       '6C' : 'e6',
                                       '6X' : 'e6',
                                       '6Z' : 'e6',
                                       }
        signal_type_dict['sbas'] = {'1C' : 'l1',
                                    '5I' : 'l5',
                                    '5Q' : 'l5',
                                    '5X' : 'l5',
                                    }
        signal_type_dict['qzss'] = {'1C' : 'l1',
                                    '1S' : 'l1',
                                    '1L' : 'l1',
                                    '1X' : 'l1',
                                    '1Z' : 'l1',
                                    '1B' : 'l1',
                                    '2S' : 'l2',
                                    '2L' : 'l2',
                                    '2X' : 'l2',
                                    '5I' : 'l5',
                                    '5Q' : 'l5',
                                    '5X' : 'l5',
                                    '5D' : 'l5',
                                    '5P' : 'l5',
                                    '5Z' : 'l5',
                                    '6S' : 'l6',
                                    '6L' : 'l6',
                                    '6X' : 'l6',
                                    '6E' : 'l6',
                                    '6Z' : 'l6',
                                    }
        signal_type_dict['beidou'] = {'2I' : 'b1',
                                      '2Q' : 'b1',
                                      '2X' : 'b1',
                                      '1D' : 'b1c',
                                      '1P' : 'b1c',
                                      '1X' : 'b1c',
                                      '1A' : 'b1c',
                                      '1N' : 'b1c',
                                      '1S' : 'b1a',
                                      '1L' : 'b1a',
                                      '1Z' : 'b1a',
                                      '5D' : 'b2a',
                                      '5P' : 'b2a',
                                      '5X' : 'b2a',
                                      '7I' : 'b2b',
                                      '7Q' : 'b2b',
                                      '7X' : 'b2b',
                                      '7D' : 'b2b',
                                      '7P' : 'b2b',
                                      '7Z' : 'b2b',
                                      '8D' : 'b2',
                                      '8P' : 'b2',
                                      '8X' : 'b2',
                                      '6I' : 'b3',
                                      '6Q' : 'b3',
                                      '6X' : 'b3',
                                      '6A' : 'b3',
                                      '6D' : 'b3a',
                                      '6P' : 'b3a',
                                      '6Z' : 'b3a',
                                      }
        signal_type_dict['irnss'] = {'5A' : 'l5',
                                     '5B' : 'l5',
                                     '5C' : 'l5',
                                     '5X' : 'l5',
                                     '9A' : 's',
                                     '9B' : 's',
                                     '9C' : 's',
                                     '9X' : 's',
                                     }
        return signal_type_dict
