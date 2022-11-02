"""Kaggle submission utilities.

"""

__authors__ = "D. Knowles"
__date__ = "31 Oct 2022"

import numpy as np

from gnss_lib_py.parsers.navdata import NavData

def prepare_submission(data, state_wls, trip_id):
    """Converts from gnss_lib_py receiver state to Kaggle submission.

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.navdata.NavData
        Instance of the NavData class. Must include ``UnixTimeMillis``.
    receiver_state : gnss_lib_py.parsers.navdata.NavData
        Estimated receiver position in latitude and longitude as an
        instance of the NavData class with the following
        rows: ``lat_*_deg``, ``lon_*_deg``.
    tripId : string
        Value for the tripId column in kaggle submission which is a
        fusion of the data and phone type

    Returns
    -------
    output : gnss_lib_py.parsers.navdata.NavData
        NavData structure ready for Kaggle submission

    """

    output = NavData()

    output["tripId"] = np.array([trip_id] * state_wls.shape[1])
    output["UnixTimeMillis"] = np.unique(data["unix_millis"])
    output["LatitudeDegrees"] = state_wls["lat_rx_deg"]
    output["LongitudeDegrees"] = state_wls["lon_rx_deg"]

    return output

def interpolate_nans(output):
    """Fills in nans.

    Parameters
    ----------
    output : gnss_lib_py.parsers.navdata.NavData
        NavData structure ready for Kaggle submission

    Returns
    -------
    output : gnss_lib_py.parsers.navdata.NavData
        NavData structure ready for Kaggle submission

    Notes
    -----
    Copied from https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array


    """

    # TODO: change to interpolate based on UnixTimeMillis

    for row in ["LatitudeDegrees","LongitudeDegrees"]:
        row_contents = output[row]

        nans, x= np.isnan(row_contents), lambda z: z.nonzero()[0]

        row_contents[nans]= np.interp(x(nans), x(~nans), row_contents[~nans])

        output[row] = row_contents

    return output
