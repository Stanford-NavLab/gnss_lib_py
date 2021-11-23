"""Functions to read data from NMEA files.

"""

__authors__ = "Shubh Gupta"
__date__ = "16 Jul 2021"

from io import BytesIO # not the gnss_lib_py/io/ modules
from datetime import datetime

import numpy as np
import pandas as pd

def _obstime(fol):
    """Convert Rinex obs (observation) time to datetime.

    Parameters
    ----------
    fol : list of strings???
        list of relevant string snippets containing the date from the
        rinex observation file header

    Returns
    -------
    result : datetime object
        converted datetime object of the provided date

    Notes
    -----
    Copied from PyGPS by Michael Hirsch and Greg Starr:
    https://github.com/gregstarr/PyGPS/blob/master/Examples/readRinexObs.py

    See the assocaited GNU Affero General Public License v3.0 here:
    https://github.com/gregstarr/PyGPS/blob/master/LICENSE.

    Python >= 3.7 supports nanoseconds. https://www.python.org/dev/peps/pep-0564/
    Python < 3.7 supports microseconds.

    """
    year = int(fol[0])
    if 80 <= year <= 99:
        year += 1900
    elif year < 80:  # because we might pass in four-digit year
        year += 2000

    result = datetime(year=year, month=int(fol[1]), day=int(fol[2]),
                      hour=int(fol[3]), minute=int(fol[4]),
                      second=int(float(fol[5])),
                      microsecond=int(float(fol[5]) % 1 * 1000000)
                      )

    return result

def read_rinex2(input_path):
    """Convert Rinex 2 file into a pandas dataframe.

    Parameters
    ----------
    input_path : string
        filepath to the

    Returns
    -------
    dsf_main : pandas dataframe
        dataframe that holds converted rinex data

    """
    STARTCOL2 = 3
    Nl = 7  # number of additional lines per record, for RINEX 2 NAV
    Lf = 19  # string length per field
    svs, raws = [], []
    dt = []
    dsf_main = pd.DataFrame()
    with open(input_path, 'r') as f:
        line = f.readline()
        ver = float(line[:9])
        assert int(ver) == 2
        if line[20] == 'N':
            svtype = 'G'  # GPS
            fields = ['SVclockBias', 'SVclockDrift', 'SVclockDriftRate',
                      'IODE', 'Crs', 'DeltaN', 'M0', 'Cuc',
                      'Eccentricity', 'Cus', 'sqrtA', 'Toe', 'Cic',
                      'Omega0', 'Cis', 'Io', 'Crc', 'omega', 'OmegaDot',
                      'IDOT', 'CodesL2', 'GPSWeek', 'L2Pflag', 'SVacc',
                      'health', 'TGD', 'IODC', 'TransTime', 'FitIntvl']
        # elif line[20] == 'G':
        #   svtype = 'R'  # GLONASS
        #   fields = ['SVclockBias', 'SVrelFreqBias', 'MessageFrameTime',
        #             'X', 'dX', 'dX2', 'health',
        #             'Y', 'dY', 'dY2', 'FreqNum',
        #             'Z', 'dZ', 'dZ2', 'AgeOpInfo']
        else:
            raise NotImplementedError(f'I do not yet handle Rinex 2 NAV {line}')

        # %% skip header, which has non-constant number of rows
        while True:
            if 'END OF HEADER' in f.readline():
                break

        # %% read data
        for ln in f:
            # format I2 http://gage.upc.edu/sites/default/files/gLAB/HTML/GPS_Navigation_Rinex_v2.11.html
            svs.append(int(ln[:2]))
            # format I2
            dt.append(_obstime([ln[3:5], ln[6:8], ln[9:11], ln[12:14],
                                ln[15:17], ln[17:20], ln[17:22]]))
            """
            now get the data as one big long string per SV
            """
            raw = ln[22:79]  # NOTE: MUST be 79, not 80 due to some files that put \n a character early!
            for _ in range(Nl):
                raw += f.readline()[STARTCOL2:79]
            # one line per SV
            raws.append(raw.replace('D', 'E'))

        # %% parse
        t = np.array([np.datetime64(t, 'ns') for t in dt])
        svu = sorted(set(svs))
        for sv in svu:
            svi = [i for i, s in enumerate(svs) if s == sv]
            tu = np.unique(t[svi])
            # Duplicates
            if tu.size != t[svi].size:
                continue

            darr = np.empty((1, len(fields)))

            ephem_ent = 3
            darr[0, :] = np.genfromtxt(BytesIO(raws[svi[ephem_ent]].encode('ascii')), delimiter=[Lf]*len(fields))

            dsf = pd.DataFrame(data=darr, index=[sv], columns=fields)
            dsf['time'] = t[svi[ephem_ent]]
            dsf['Svid'] = sv

            # print(dsf['time'], dsf.GPSWeek)
            dsf_main = pd.concat([dsf_main, dsf])
    return dsf_main
