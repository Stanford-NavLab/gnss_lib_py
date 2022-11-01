"""Kaggle submission utilities.

"""

__authors__ = "D. Knowles"
__date__ = "31 Oct 2022"

import numpy as np

def prepare_submission(data, state_wls):
    """Converts from gnss_lib_py receiver state to Kaggle submission.

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Instance of the NavData class. Must include ``UnixTimeMillis``.
    receiver_state : gnss_lib_py.parsers.navdata.NavData
        Estimated receiver position in latitude and longitude as an
        instance of the NavData class with the following
        rows: ``lat_*_deg``, ``lon_*_deg``.

    """
    output = state_wls.copy()

    # TODO: search for correct lat/lon
    output.rename({"lat_rx_deg" : "LatitudeDegrees",
                   "lon_rx_deg" : "LongitudeDegrees",
                  }, inplace=True)
    # TODO: delete all unnecessary rows no matter what they are
    output.remove(rows=['gps_millis', 'x_rx_m', 'y_rx_m', 'z_rx_m',
                        'b_rx_m', 'alt_rx_deg'], inplace=True)

    output["UnixTimeMillis"] = np.unique(data["unix_millis"])

    return output
