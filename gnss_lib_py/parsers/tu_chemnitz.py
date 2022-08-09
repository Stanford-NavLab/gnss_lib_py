"""Functions to process TU Chemnitz SmartLoc dataset measurements.

"""

__authors__ = "Derek Knowles"
__date__ = "09 Aug 2022"

import numpy as np

from gnss_lib_py.parsers.navdata import NavData


class TUChemnitzRaw(NavData):
    """Class handling raw measurements from TU Chemnitz dataset [1]_.

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
        """TU Chemniz raw specific loading and preprocessing.

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

        self["gnss_id"] = np.array([x.lower() for x in self["gnss_id"][0]], dtype=object)


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
