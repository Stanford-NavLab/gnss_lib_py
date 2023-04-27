"""Functions to download, save and process satellite ephemeris files.

The Ephemeris Manager provides broadcast ephemeris for specific
satellites at a specific timestep. The EphemerisManager class should be
initialized and then the ``get_ephemeris`` function can be used to
retrieve ephemeris for specific satellites. ``get_ephemeris`` returns
the most recent broadcast ephemeris for the provided list of satellites
that was broadcast BEFORE the provided timestamp. For example GPS daily
ephemeris files contain data at a two hour frequency, so if the
timestamp provided is 5am, then ``get_ephemeris`` will return the 4am
data but not 6am. If provided a timestamp between midnight and 2am then
the ephemeris from around midnight (might be the day before) will be
provided. If no list of satellites is provided, then ``get_ephemeris``
will return data for all satellites.

When multiple observations are provided for the same satellite and same
timestep, the Ephemeris Manager will only return the first instance.
This is applicable when requesting ephemeris for multi-GNSS for the
current day. Same-day multi GNSS data is pulled from  same day. For
same-day multi-GNSS from https://igs.org/data/ which often has multiple
observations.

"""

__authors__ = "Shubh Gupta, Ashwin Kanhere"
__date__ = "13 July 2021"

import os
import shutil
import gzip
import ftplib
from ftplib import FTP_TLS, FTP
from datetime import datetime, timezone

import unlzw3
import georinex
import numpy as np
import pandas as pd

import gnss_lib_py.utils.constants as consts
from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.utils.time_conversions import datetime_to_gps_millis, tzinfo_to_utc


class EphemerisManager():
    """Download, store and process ephemeris files

    Attributes
    ----------
    data_directory : string
        Directory to store/read ephemeris files
    data : pd.Dataframe
        Ephemeris parameters
    leapseconds : int
        Leap seconds to add to UTC time to get GPS time
    verbose : bool
        If true, prints debugging statements.

    Notes
    -----
    Class code taken from https://github.com/johnsonmitchelld/gnss-analysis/blob/main/gnssutils/ephemeris_manager.py

    The associated license is copied below:

    BSD 3-Clause License

    Copyright (c) 2021, Mitchell D Johnson
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
       list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    """
    def __init__(self, data_directory=os.path.join(os.getcwd(), 'data', 'ephemeris'),
                 verbose=False):
        self.data_directory = data_directory
        nasa_dir = os.path.join(data_directory, 'nasa')
        igs_dir = os.path.join(data_directory, 'igs')
        os.makedirs(nasa_dir, exist_ok=True)
        os.makedirs(igs_dir, exist_ok=True)
        self.data = None
        self.leapseconds = None
        self.verbose = verbose

    def get_ephemeris(self, timestamp, satellites=None):
        """Return ephemeris DataFrame for satellites input.

        The Ephemeris Manager provides broadcast ephemeris for specific
        satellites at a specific timestep. The EphemerisManager class
        should be initialized and then the ``get_ephemeris`` function
        can be used to retrieve ephemeris for specific satellites.
        ``get_ephemeris`` returns the most recent broadcast ephemeris
        for the provided list of satellites that was broadcast BEFORE
        the provided timestamp. For example GPS daily ephemeris files
        contain data at a two hour frequency, so if the timestamp
        provided is 5am, then ``get_ephemeris`` will return the 4am data
        but not 6am. If provided a timestamp between midnight and 2am
        then the ephemeris from around midnight (might be the day
        before) will be provided. If no list of satellites is provided,
        then ``get_ephemeris`` will return data for all satellites.

        When multiple observations are provided for the same satellite
        and same timestep, the Ephemeris Manager will only return the
        first instance. This is applicable when requesting ephemeris for
        multi-GNSS for the current day. Same-day multi GNSS data is
        pulled from  same day. For same-day multi-GNSS from
        https://igs.org/data/ which often has multiple observations.

        Parameters
        ----------
        timestamp : datetime.datetime
            Ephemeris data is returned for the timestamp day and
            includes all broadcast ephemeris whose broadcast timestamps
            happen before the given timestamp variable. Timezone should
            be added manually and is interpreted as UTC if not added.
        satellites : List
            List of satellite IDs as a string, for example ['G01','E11',
            'R06']. Defaults to None which returns get_ephemeris for
            all satellites.

        Returns
        -------
        data : gnss_lib_py.parsers.navdata.NavData
            ephemeris entries corresponding to timestamp

        Notes
        -----
        The Galileo week ``GALWeek`` is identical to the GPS Week
        ``GPSWeek``. See http://acc.igs.org/misc/rinex304.pdf page A26

        """
        systems = EphemerisManager.get_constellations(satellites)
        # add UTC timezone if datatime os offset-naive
        timestamp = tzinfo_to_utc(timestamp)
        if not isinstance(self.data, pd.DataFrame):
            same_day = (datetime.now(timezone.utc) - timestamp).days <= 0
            self.load_data(timestamp, systems, same_day)
        data = self.data
        if satellites is not None:
            data = data.loc[data['sv'].isin(satellites)]
        time_cropped_data = data.loc[data['time'] < timestamp]
        time_cropped_data = time_cropped_data.sort_values('time').groupby(
            'sv').last().drop(labels = 'index', axis = 'columns')
        if satellites is not None and len(time_cropped_data) < len(satellites):
            # if no data available for the given day, try looking at the
            # previous day, may occur when a time near to midnight
            # is provided. For example, 12:01am
            if len(time_cropped_data) != 0:
                satellites = list(set(satellites) - set(time_cropped_data.index))
            systems = EphemerisManager.get_constellations(satellites)
            prev_day_timestamp = datetime(year=timestamp.year,
                                          month=timestamp.month,
                                          day=timestamp.day - 1,
                                          hour=23,
                                          minute=59,
                                          second=59,
                                          microsecond=999999,
                                          tzinfo=timezone.utc,
                                          )
            self.load_data(prev_day_timestamp, systems, False)
            prev_data = self.data
            if satellites is not None:
                prev_data = prev_data.loc[prev_data['sv'].isin(satellites)]
            prev_data = prev_data.sort_values('time').groupby(
                'sv').last().drop(labels = 'index', axis = 'columns')
            data = pd.concat((time_cropped_data,prev_data))
        else:
            data = time_cropped_data
        data['Leap Seconds'] = self.leapseconds
        # Convert data DataFrame to NavData instance
        # Move sv to DataFrame columns, reset index
        data = data.reset_index()
        # Replace datetime with gps_millis
        gps_millis = [datetime_to_gps_millis(df_row['time']) \
                        for _, df_row in data.iterrows()]
        data['gps_millis'] = gps_millis
        data = data.drop(columns=['time'])
        data = data.rename(columns={"sv":"sv_id"})
        if "GPSWeek" in data.columns:
            data = data.rename(columns={"GPSWeek":"gps_week"})
            if "GALWeek" in data.columns:
                data["gps_week"] = np.where(pd.isnull(data["gps_week"]),
                                                      data["GALWeek"],
                                                      data["gps_week"])
        elif "GALWeek" in data.columns:
            data = data.rename(columns={"GALWeek":"gps_week"})
        if len(data) == 0:
            raise RuntimeError("No ephemeris data available for the " \
                             + "given satellites")
        data_navdata = NavData(pandas_df=data)
        data_navdata['gnss_sv_id'] = data_navdata['sv_id']
        gnss_chars = [sv_id[0] for sv_id in np.atleast_1d(data_navdata['sv_id'])]
        gnss_nums = [sv_id[1:] for sv_id in np.atleast_1d(data_navdata['sv_id'])]
        gnss_id = [consts.CONSTELLATION_CHARS[gnss_char] for gnss_char in gnss_chars]
        data_navdata['gnss_id'] = np.asarray(gnss_id)
        data_navdata['sv_id'] = np.asarray(gnss_nums, dtype=int)
        return data_navdata

    def load_data(self, timestamp, constellations=None, same_day=False):
        """Load ephemeris into class instance

        Parameters
        ----------
        timestamp : datetime.datetime
            Ephemeris data is returned for the timestamp day and
            includes all broadcast ephemeris whose broadcast timestamps
            happen before the given timestamp variable. Timezone should
            be added manually and is interpreted as UTC if not added.
        constellations : Set
            Set of satellites For example, set({"G","R","E"}).
        same_day : bool
            Whether or not ephemeris is for same-day aquisition.

        """
        filepaths = EphemerisManager.get_filepaths(timestamp)
        data_list = []

        if constellations == None:
            for fileinfo in filepaths.values():
                data = self.get_ephemeris_dataframe(fileinfo,
                                                    constellations)
                data_list.append(data)
        else:
            legacy_systems = set(['G', 'R'])
            legacy_systems_only = len(constellations - legacy_systems) == 0
            if not same_day:
                if legacy_systems_only:
                    if 'G' in constellations:
                        data_list.append(self.get_ephemeris_dataframe(
                            filepaths['nasa_daily_gps'],
                            constellations=None))
                    if 'R' in constellations:
                        data_list.append(self.get_ephemeris_dataframe(
                            filepaths['nasa_daily_glonass'],
                            constellations=None))
                else:
                    data_list.append(self.get_ephemeris_dataframe(
                        filepaths['nasa_daily_combined'],
                        constellations))
            else:
                if legacy_systems_only and 'G' in constellations:
                    data_list.append(self.get_ephemeris_dataframe(
                        filepaths['nasa_daily_gps'],
                        constellations=None))
                else:
                    data_list.append(self.get_ephemeris_dataframe(
                        filepaths['bkg_daily_combined'],
                        constellations))

        data = pd.DataFrame()
        for new_data in data_list:
            data = pd.concat((data,new_data), ignore_index=True)

        data.reset_index(inplace=True)
        data.sort_values('time', inplace=True, ignore_index=True)
        self.data = data

    def get_ephemeris_dataframe(self, fileinfo, constellations=None):
        """Load/download ephemeris files and process into DataFrame

        Parameters
        ----------
        fileinfo : dict
            Filenames for ephemeris with ftp server and constellation details

        constellations : Set
            Set of satellites {"ConstIDSVID"}

        Returns
        -------
        data : pd.DataFrame
            Parsed ephemeris DataFrame
        """
        filepath = fileinfo['filepath']
        url = fileinfo['url']
        directory = os.path.split(filepath)[0]
        filename = os.path.split(filepath)[1]
        if url == 'igs-ftp.bkg.bund.de':
            dest_filepath = os.path.join(self.data_directory, 'igs', filename)
        else:
            dest_filepath = os.path.join(self.data_directory, 'nasa', filename)
        decompressed_filename = os.path.splitext(dest_filepath)[0]
        if not os.path.isfile(decompressed_filename): # pragma: no cover
            self.retrieve_file(url, directory, filename,
                               dest_filepath)
        if not self.leapseconds:
            self.leapseconds = EphemerisManager.load_leapseconds(
                decompressed_filename)
        if constellations is not None:
            data = georinex.load(decompressed_filename,
                                 use=constellations,
                                 verbose=self.verbose).to_dataframe()
        else:
            data = georinex.load(decompressed_filename,
                                 verbose=self.verbose).to_dataframe()
        data.dropna(how='all', inplace=True)
        data.reset_index(inplace=True)
        data['source'] = decompressed_filename
        data['t_oc'] = pd.to_numeric(data['time'] - datetime(1980, 1, 6, 0, 0, 0))
        #TODO: Use a constant for the time of GPS clock start
        data['t_oc']  = 1e-9 * data['t_oc'] - consts.WEEKSEC * np.floor(1e-9 * data['t_oc'] / consts.WEEKSEC)
        data['time'] = data['time'].dt.tz_localize('UTC')
        data.rename(columns={'M0': 'M_0', 'Eccentricity': 'e', 'Toe': 't_oe', 'DeltaN': 'deltaN', 'Cuc': 'C_uc', 'Cus': 'C_us',
                             'Cic': 'C_ic', 'Crc': 'C_rc', 'Cis': 'C_is', 'Crs': 'C_rs', 'Io': 'i_0', 'Omega0': 'Omega_0'}, inplace=True)
        return data


    @staticmethod
    def get_filetype(timestamp):
        """Get file extension of IGS file based on timestamp

        Parameters
        ----------
        timestamp : datetime.datetime
            Time of clock

        Returns
        -------
        extension : string
            Extension of compressed ephemeris file
        """
        # IGS switched from .Z to .gz compression format on December 1st, 2020
        if timestamp >= datetime(2020, 12, 1, 0, 0, 0, tzinfo=timezone.utc):
            extension = '.gz'
        else:
            extension = '.Z'
        return extension

    @staticmethod
    def load_leapseconds(filename):
        """Read leapseconds from ephemeris file

        Parameters
        ----------
        filename : string
            Ephemeris filename

        Returns
        -------
        read_lp_sec : int or None
            Leap seconds read from file

        """
        with open(filename) as f:
            for line in f:
                if 'LEAP SECONDS' in line:
                    read_lp_sec = int(line.split()[0])
                    return read_lp_sec
                if 'END OF HEADER' in line:
                    return None

        return None

    @staticmethod
    def get_constellations(satellites):
        """Convert list of satellites to set

        Parameters
        ----------
        satellites : List
            List of satellites of form [ConstIDSVID]

        Returns
        -------
        systems : Set or None
            Set representation of satellites for which ephemeris is needed
        """
        if isinstance(satellites, list):
            systems = set()
            for sat in satellites:
                systems.add(sat[0])
            return systems

        return None

    def retrieve_file(self, url, directory, filename, dest_filepath):
        """Copy ephemeris file from FTP filepath to local directory.

        Also decompresses file.

        Parameters
        ----------
        url : String
            FTP server location

        directory : String
            Directory where ephemeris files are stored on the FTP server

        filename : String
            Filename in which ephemeris files are stored (both locally and globally)

        dest_filepath : String
            Directory where downloaded ephemeris files are stored locally

        """

        secure = bool(url == 'gdc.cddis.eosdis.nasa.gov')

        if self.verbose:
            print('Retrieving ' + directory + '/' + filename + ' from ' + url)
        ftp = self.connect(url, secure)
        src_filepath = directory + '/' + filename
        try:
            with open(dest_filepath, 'wb') as handle:
                ftp.retrbinary(
                    'RETR ' + src_filepath, handle.write)
        except ftplib.error_perm as err:
            os.remove(dest_filepath)
            raise ftplib.error_perm(str(err) + ' Failed to retrieve ' \
                                  + src_filepath + ' from ' + url)

        ftp.quit()
        if ftp is not None: # try closing if still active
            ftp.close()
        self.decompress_file(dest_filepath)

    def decompress_file(self, filepath):
        """Decompress downloaded file ephemeris file in same destination location

        Parameters
        ----------
        filepath : String
            Local filepath where the compressed ephemeris file is stored

        """
        extension = os.path.splitext(filepath)[1]
        decompressed_path = os.path.splitext(filepath)[0]
        if extension == '.gz':
            with gzip.open(filepath, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        elif extension == '.Z':
            with open(filepath, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    f_out.write(unlzw3.unlzw(f_in.read()))
        os.remove(filepath)

    def connect(self, url, secure):
        """Connect to given FTP server

        Parameters
        ----------
        url : String
            URL of FTP server where ephemeris files are stored

        secure : Bool
            Flag for secure FTP connection

        Returns
        -------
        ftp : FTP_TLS
            FTP connection object
        """
        if secure:
            ftp = FTP_TLS(url)
            ftp.login()
            ftp.prot_p()
        else:
            ftp = FTP(url)
            ftp.login()
        return ftp

    @staticmethod
    def get_filepaths(timestamp):
        """Generate filepaths for all ephemeris files

        Parameters
        ----------
        timestamp : datetime.datetime
            Time of clock

        Returns
        -------
        filepaths : Dict
            Dictionary of dictionaries containing filepath and directory for ephemeris files
        """
        timetuple = timestamp.timetuple()
        extension = EphemerisManager.get_filetype(timestamp)
        filepaths = {}

        directory = 'gnss/data/daily/' + str(timetuple.tm_year) + '/brdc/'
        filename = 'BRDC00IGS_R_' + \
            str(timetuple.tm_year) + \
            str(timetuple.tm_yday).zfill(3) + '0000_01D_MN.rnx.gz'
        filepaths['nasa_daily_combined'] = {
            'filepath': directory + filename, 'url': 'gdc.cddis.eosdis.nasa.gov'}

        filename = 'brdc' + str(timetuple.tm_yday).zfill(3) + \
            '0.' + str(timetuple.tm_year)[-2:] + 'n' + extension
        filepaths['nasa_daily_gps'] = {
            'filepath': directory + filename, 'url': 'gdc.cddis.eosdis.nasa.gov'}

        filename = 'brdc' + str(timetuple.tm_yday).zfill(3) + \
            '0.' + str(timetuple.tm_year)[-2:] + 'g' + extension
        filepaths['nasa_daily_glonass'] = {
            'filepath': directory + filename, 'url': 'gdc.cddis.eosdis.nasa.gov'}

        directory = '/IGS/BRDC/' + \
            str(timetuple.tm_year) + '/' + \
            str(timetuple.tm_yday).zfill(3) + '/'
        filename = 'BRDC00WRD_S_' + \
            str(timetuple.tm_year) + \
            str(timetuple.tm_yday).zfill(3) + '0000_01D_MN.rnx.gz'
        filepaths['bkg_daily_combined'] = {
            'filepath': directory + filename, 'url': 'igs-ftp.bkg.bund.de'}

        return filepaths
