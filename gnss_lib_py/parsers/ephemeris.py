"""Functions to download, save and process satellite ephemeris files.

"""

__authors__ = "Shubh Gupta, Ashwin Kanhere"
__date__ = "13 July 2021"

import os
import shutil
import gzip
import ftplib
from ftplib import FTP_TLS, FTP
from datetime import datetime, timedelta, timezone

import unlzw3
import georinex
import numpy as np
import pandas as pd

import gnss_lib_py.utils.constants as consts


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
    def __init__(self, data_directory=os.path.join(os.getcwd(), 'data', 'ephemeris')):
        self.data_directory = data_directory
        nasa_dir = os.path.join(data_directory, 'nasa')
        igs_dir = os.path.join(data_directory, 'igs')
        os.makedirs(nasa_dir, exist_ok=True)
        os.makedirs(igs_dir, exist_ok=True)
        self.data = None
        self.leapseconds = None

    def get_ephemeris(self, timestamp, satellites):
        """Return ephemeris DataFrame for satellites input

        Parameters
        ----------
        timestamp : datetime.datetime
            Time of clock
        satellites : List
            List of satellites ['Const_IDSVID']

        Returns
        -------
        data : pd.DataFrame
            DataFrame containing ephemeris entries corresponding to timestamp

        """
        systems = EphemerisManager.get_constellations(satellites)
        if not isinstance(self.data, pd.DataFrame):
            self.load_data(timestamp, systems)
        data = self.data
        if satellites:
            data = data.loc[data['sv'].isin(satellites)]
        data = data.loc[data['time'] < timestamp]
        data = data.sort_values('time').groupby(
            'sv').last().drop(labels = 'index', axis = 'columns')
        data['Leap Seconds'] = self.leapseconds
        return data

    def get_leapseconds(self, timestamp):
        """Output saved leapseconds

        Returns
        -------
        lp_seconds : float
            Leap seconds between GPS and UTC time
        """
        lp_seconds = self.leapseconds
        return lp_seconds

    def load_data(self, timestamp, constellations=None):
        """Load ephemeris into class instance

        Parameters
        ----------

        timestamp : datetime.datetime
            Time of clock

        constellations : Set
            Set of satellites {"ConstIDSVID"}

        """
        filepaths = EphemerisManager.get_filepaths(timestamp)
        data_list = []
        timestamp_age = datetime.now(timezone.utc) - timestamp
        if constellations == None:
            for fileinfo in filepaths.values():
                data = self.get_ephemeris_dataframe(fileinfo)
                data_list.append(data)
        else:
            legacy_systems = set(['G', 'R'])
            legacy_systems_only = len(constellations - legacy_systems) == 0
            if timestamp_age.days > 0:
                if legacy_systems_only:
                    data_list.append(self.get_ephemeris_dataframe(
                        filepaths['nasa_daily_gps']))
                    if 'R' in constellations:
                        data_list.append(self.get_ephemeris_dataframe(
                            filepaths['nasa_daily_glonass']))
                else:
                    data_list.append(self.get_ephemeris_dataframe(
                        filepaths['nasa_daily_combined']))
            else:
                data_list.append(self.get_ephemeris_dataframe(
                    filepaths['nasa_daily_gps']))
                if not legacy_systems_only:
                    data_list.append(self.get_ephemeris_dataframe(
                        filepaths['bkg_daily_combined']))

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
        if url == 'igs.bkg.bund.de':
            dest_filepath = os.path.join(self.data_directory, 'igs', filename)
        else:
            dest_filepath = os.path.join(self.data_directory, 'nasa', filename)
        decompressed_filename = os.path.splitext(dest_filepath)[0]
        if not os.path.isfile(decompressed_filename):
            if url == 'gdc.cddis.eosdis.nasa.gov':
                secure = True
            else:
                secure = False
            try:
                self.retrieve_file(url, directory, filename,
                                   dest_filepath, secure)
                self.decompress_file(dest_filepath)
            except ftplib.error_perm as err:
                print('ftp error')
                return pd.DataFrame()
        if not self.leapseconds:
            self.leapseconds = EphemerisManager.load_leapseconds(
                decompressed_filename)
        if constellations:
            data = georinex.load(decompressed_filename,
                                 use=constellations).to_dataframe()
        else:
            data = georinex.load(decompressed_filename).to_dataframe()
        data.dropna(how='all', inplace=True)
        data.reset_index(inplace=True)
        data['source'] = decompressed_filename
        data['t_oc'] = pd.to_numeric(data['time'] - datetime(1980, 1, 6, 0, 0, 0))
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
        if type(satellites) is list:
            systems = set()
            for sat in satellites:
                systems.add(sat[0])
            return systems
        else:
            return None

    @staticmethod
    def calculate_toc(timestamp):
        """I think this is equivalent of datetime_to_tow()
        #TODO: See if this function is needed or can be deleted
        """
        pass

    def retrieve_file(self, url, directory, filename, dest_filepath, secure=False):
        """Copy ephemeris file from FTP filepath to local directory

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

        secure : Bool
            Whether to make secure FTP connection

        """
        print('Retrieving ' + directory + '/' + filename + ' from ' + url)
        ftp = self.connect(url, secure)
        src_filepath = directory + '/' + filename
        try:
            with open(dest_filepath, 'wb') as handle:
                ftp.retrbinary(
                    'RETR ' + src_filepath, handle.write)
        except ftplib.error_perm as err:
            print('Failed to retrieve ' + src_filepath + ' from ' + url)
            print(err)
            os.remove(dest_filepath)
            raise ftplib.error_perm

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

    def listdir(self, url, directory, secure):
        """Display files on server that match input filename

        Parameters
        ----------
        url : String
            URL of FTP server

        directory : String
            Directory in FTP server where relevant ephemeris files are stored
        """
        # TODO: Function not called anywhere, consider folding into existing function calls or deleting
        ftp = self.connect(url, secure)
        dirlist = ftp.nlst(directory)
        dirlist = [x for x in dirlist]
        print(dirlist)

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
            'filepath': directory + filename, 'url': 'igs.bkg.bund.de'}

        return filepaths


if __name__ == '__main__':
    repo = EphemerisManager()
    target_time = datetime(2021, 1, 9, 12, 0, 0, tzinfo=timezone.utc)
    data = repo.get_ephemeris(target_time, ['G01', 'G03'])
