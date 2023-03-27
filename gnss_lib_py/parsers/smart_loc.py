"""Functions to process TU Chemnitz SmartLoc dataset measurements.

"""

__authors__ = "Derek Knowles"
__date__ = "09 Aug 2022"

import numpy as np

from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.utils.time_conversions import tow_to_gps_millis


class SmartLocRaw(NavData):
    """Class handling raw measurements from SmartLoc dataset [1]_.

    The SmartLoc dataset is a GNSS dataset from TU Chemnitz
    Dataset is available on their website [2]_. Inherits from NavData().

    References
    ----------
    .. [1] Reisdorf, Pierre, Tim Pfeifer, Julia Bressler, Sven Bauer,
           Peter Weissig, Sven Lange, Gerd Wanielik and Peter Protzel.
           The Problem of Comparable GNSS Results – An Approach for a
           Uniform Dataset with Low-Cost and Reference Data. Vehicular.
           2016.
    .. [2] https://www.tu-chemnitz.de/projekt/smartLoc/gnss_dataset.html.en#Home


    """
    def __init__(self, input_path):
        """TU Chemnitz raw specific loading and preprocessing.

        Should input path to RXM-RAWX.csv file.

        Parameters
        ----------
        input_path : string
            Path to measurement csv file

        """

        super().__init__(csv_path=input_path, sep=";")

    def postprocess(self):
        """TU Chemnitz raw specific postprocessing

        """

        # convert gnss_id to lowercase as per standard naming convention
        self["gnss_id"] = np.array([x.lower() for x in self["gnss_id"]],
                                    dtype=object)

        # create gps_millis row from gps week and time of week
        self["gps_millis"] = [tow_to_gps_millis(*x) for x in
                              zip(self["gps_week"],self["gps_tow"])]


    @staticmethod
    def _row_map():
        """Map of column names from loaded to gnss_lib_py standard

        Returns
        -------
        row_map : Dict
            Dictionary of the form {old_name : new_name}
        """

        row_map = {'GPS week number (week) [weeks]' : 'gps_week',
                   'Measurement time of week (rcvTow) [s]' : 'gps_tow',
                   'Pseudorange measurement (prMes) [m]' : 'raw_pr_m',
                   'GNSS identifier (gnssId) []' : 'gnss_id',
                   'Satellite identifier (svId) []' : 'sv_id',
                   'Carrier-to-noise density ratio (cno) [dbHz]' : 'cn0_dbhz',
                   'Estimated pseudorange measurement standard deviation (prStdev) [m]' : 'raw_pr_sigma_m',
                   }
        return row_map
