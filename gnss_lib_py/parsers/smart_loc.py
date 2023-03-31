"""Functions to process TU Chemnitz SmartLoc dataset measurements.

"""

__authors__ = "Derek Knowles"
__date__ = "09 Aug 2022"

import numpy as np

from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.utils.coordinates import LocalCoord, geodetic_to_ecef
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
                   'Doppler measurement (doMes) [Hz]' : 'doppler_hz',
                   'Estimated Doppler measurement standard deviation (doStdev) [Hz]' \
                   : 'doppler_sigma_hz',
                   'Longitude (GT Lon) [deg]' : 'lon_rx_gt_deg',
                   'Longitude Cov (GT Lon) [deg]' : 'lon_rx_gt_sigma_deg',
                   'Latitude (GT Lat) [deg]' : 'lat_rx_gt_deg',
                   'Latitude Cov (GT Lat) [deg]' : 'lat_rx_gt_sigma_deg',
                   'Height above ellipsoid (GT Height) [m]' : 'alt_rx_gt_m',
                   'Height above ellipsoid Cov (GT Height) [m]' : 'alt_rx_gt_sigma_m'

                   }
        return row_map

def remove_nlos(smartloc_raw):
    """Remove NLOS measurements from SmartLoc data instance.

    Parameters
    ----------
    smartloc_raw : gnss_lib_py.parsers.navdata.NavData
        Instance of NavData containing SmartLoc data

    Returns
    -------
    smartloc_los : gnss_lib_py.parsers.navdata.NavData
        Instance of NavData containing only LOS labeled measurements

    """
    smartloc_los = smartloc_raw.where('NLOS (0 == no, 1 == yes, # == No Information)',
                        1, 'eq')
    return smartloc_los

def calculate_gt_ecef(smartloc_raw):
    """Calculate ground truth positions in ECEF.

    Parameters
    ----------
    smartloc_raw : gnss_lib_py.parsers.navdata.NavData
        Instance of NavData containing SmartLoc data

    Returns
    -------
    smartloc_los : gnss_lib_py.parsers.navdata.NavData
        Instance of NavData containing GT ECEF positions in meters.

    """
    llh = smartloc_raw[['lat_rx_gt_deg', 'lon_rx_gt_deg', 'alt_rx_gt_m']]
    rx_ecef = geodetic_to_ecef(llh)
    smartloc_raw['x_rx_gt_m'] = rx_ecef[0, :]
    smartloc_raw['y_rx_gt_m'] = rx_ecef[1, :]
    smartloc_raw['z_rx_gt_m'] = rx_ecef[2, :]
    return smartloc_raw

def calculate_gt_vel(smartloc_raw):
    """Calculate ground truth velocity and acceleration in ECEF.

    Parameters
    ----------
    smartloc_raw : gnss_lib_py.parsers.navdata.NavData
        Instance of NavData containing SmartLoc data

    Returns
    -------
    smartloc_los : gnss_lib_py.parsers.navdata.NavData
        Instance of NavData containing GT ECEF velocity and acceleration
        in meters per second and meters per second^2.

    """
    llh_origin = smartloc_raw[['lat_rx_gt_deg', 'lon_rx_gt_deg', 'alt_rx_gt_m'] , 0]
    ned_frame = LocalCoord.from_geodetic(llh_origin.reshape([3,1]))
    vel_acc = {'vx_rx_gt_mps' : [],
                'vy_rx_gt_mps' : [],
                'vz_rx_gt_mps' : [],
                'ax_rx_gt_mps2' : [],
                'ay_rx_gt_mps2' : [],
                'az_rx_gt_mps2' : []}
    for _, _, measure_frame in smartloc_raw.loop_time('gps_millis', \
                                                        delta_t_decimals = -2):
        yaw = measure_frame['Heading (0° = East, counterclockwise) - (GT Heading) [rad]', 0]
        vel_gt = measure_frame['Velocity (GT Velocity) [m/s]', 0]
        acc_gt = measure_frame['Acceleration (GT Acceleration) [ms^2]', 0]
        vel_ned = np.array([[np.sin(yaw)*vel_gt], [np.cos(yaw)*vel_gt], [0.]])
        acc_ned = np.array([[np.sin(yaw)*acc_gt], [np.cos(yaw)*acc_gt], [0.]])
        vel_ecef = ned_frame.ned_to_ecefv(vel_ned)
        acc_ecef = ned_frame.ned_to_ecefv(acc_ned)
        vel_acc['vx_rx_gt_mps'].extend(np.repeat(vel_ecef[0, 0], len(measure_frame)))
        vel_acc['vy_rx_gt_mps'].extend(np.repeat(vel_ecef[1, 0], len(measure_frame)))
        vel_acc['vz_rx_gt_mps'].extend(np.repeat(vel_ecef[2, 0], len(measure_frame)))
        vel_acc['ax_rx_gt_mps2'].extend(np.repeat(acc_ecef[0, 0], len(measure_frame)))
        vel_acc['ay_rx_gt_mps2'].extend(np.repeat(acc_ecef[1, 0], len(measure_frame)))
        vel_acc['az_rx_gt_mps2'].extend(np.repeat(acc_ecef[2, 0], len(measure_frame)))
    for row, values in vel_acc.items():
        smartloc_raw[row] = values
    return smartloc_raw

