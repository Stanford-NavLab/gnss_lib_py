"""Functions to download Rinex, SP3, and CLK ephemeris files.

Rinex Navigation
----------------

Rinex navigation files are pulled from one of five sources. More about
rinex files can be found in the CDDIS documentation [1]_.

If rinex for the current day is requested or if rinex for the
previous is requested and it is too early in the UTC day for
yesterday's combined rinex file to be uploaded, then the
BRDC00WRD_S rinex 3 file is downloaded from IGS and includes:
GPS+GLO+GAL+BDS+QZSS+SBAS.

If only GPS is requested, then the brdc<dddd>.<yy>n rinex 2 file is
downloaded from CDDIS and includes: GPS.

If only Glonass is requested, then the brdc<dddd>.<yy>g rinex 2 file
is downloaded from CDDIS and includes: GLO.

If the request is for multi-gnss and for a date before Nov. 25, 2019,
but after Jan 1, 2013 (the start of multi-gnss) then the BRDM00DLR_R
rinex 3 file is downloaded from CDDIS. If multi-gnss is requested
for Nov. 25, 2019 or later, then the BRDM00DLR_S rinex 3 file is
downloaded from CDDIS. These multi-gnss files include:
GPS+GLO+GAL+BDS+QZSS+IRNSS+SBAS. For more information on multi-gnss
combined rinex navigation files, see MGEX documentation [2]_.

SP3 and CLK within Two Weeks
----------------------------
SP3 and CLK files are obtained from CDDIS and produced by either the
Center for Orbit Determination in Europe (CODE), GeoForschungsZentrum
Potsdam (GFZ) or Wuhan University. Products are available through the
MGEX data program [4]_. Details on the MGEX precise orbit and clock
products can be found on the IGS website [4]_.

If the SP3 or CLK date requested is within the three days, then the
rapid solution from CODE is downloaded (COD0OPSRAP). The CODE rapid
solution includes: GPS+GLO+GAL.

If the SP3 or CLK date requested is within the last two weeks, then the
rapid solution from GFZ is downloaded (GFZ0MGXRAP). The GFZ rapid
solution became available starting GPS week 2038 or Jan 27, 2019. The
GFZ rapid solution includes: GPS+GLO+GAL+BDS+QZS. More about the GFZ
solution can be found on their MGEX website [8]_.

SP3 Older than Two Weeks
------------------------
If the SP3 date requested is more than two weeks previous to the
current date and GPS week 1962 or later, then the CODE final solution
is downloaded (COD0MGXFIN). The CODE final SP3 solutions became
available starting GPS week 1962 or Aug 13, 2017. The CODE final
solution includes: GPS+GLO+GAL+BDS+QZS. More about the CODE solution can
be found in their papers [6]_ [7]_.

If the SP3 date requested is between GPS week 1690 and 1961 or between
May 25, 2012 and Aug 12, 2017, then the CODE 'com' short name solution
is downloaded. More about the CODE 'com' solution can be found in their
papers [9]_ [10]_.

CLK Older than Two Weeks
------------------------
If the CLK date requested is more than two weeks previous to the
current date and GPS week 2113 or later, then the CODE final solution
is downloaded (COD0MGXFIN). The CODE final CLK solutions became
available starting GPS week 2113 or July 5th, 2020. The CODE final
solution includes: GPS+GLO+GAL+BDS+QZS. More about the CODE solution can
be found in their papers [6]_ [7]_.

If the CLK date requested is between GPS week 2034 and 2112 or between
Jan 1, 2019 and July 4th, 2020, then the Wuhan University final
solution is downloaded (WUM0MGXFIN). More about the WUM0MGXFIN solution
can be found in their documention [11]_.

If the CLK date requested is week GPS week 2034 and between Dec 30, 2018
and Dec 31, 2018, then the GFZ short name solution is downloaded (gbm).
More about the GFZ 'gbm' solution can be found in their
documentation [13]_ [14]_.

If the CLK date requested is between GPS week 1962 and 2033 or between
Aug 13, 2017 and Dec 29th, 2018 then the Wuhan University short name
solution is downloaded (wum). More about the "wum" solution can be found
in their paper [12]_.

If the CLK date requested is between GPS week 1710 and 1961 or between
Oct 14, 2012 and Aug 12, 2017, then the CODE 'com' short name solution
is downloaded. More about the CODE 'com' solution can be found in their
papers [9]_ [10]_.

IGS Resources
-------------

IGS network station information can be found at [3]_. IGS files can be
viewed online using their file browser [5]_.

References
----------
.. [1] https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/daily_30second_data.html
.. [2] https://igs.org/mgex/data-products/#bce
.. [3] https://network.igs.org/
.. [4] https://igs.org/mgex/data-products/#orbit_clock
.. [5] https://igs.bkg.bund.de/browseFiles
.. [6] http://dx.doi.org/10.1016/j.asr.2020.04.038
.. [7] http://dx.doi.org/10.7892/boris.75882.3
.. [8] https://www.gfz-potsdam.de/en/section/space-geodetic-techniques/projects/mgex
.. [9] http://doi.org/10.1007/1345_2015_161
.. [10] http://doi.org/10.1007/s00190-016-0968-8
.. [11] https://files.igs.org/pub/center/analysis/WUM0MGXFIN.acn
.. [12] http://dx.doi.org/10.1007/s00190-015-0862-9
.. [13] https://files.igs.org/pub/resource/pubs/workshop/2014/Workshop%202014%20-%20PS11%20-%20Deng%20-%202226%20-%20Precise%20orbit%20determination%20of%20Beidou%20Satellites%20at%20GFZ.pdf
.. [14] http://doi.org/10.1007/1345_2015_120

"""

# CLK: 2020, 7, 2

__authors__ = "Shubh Gupta, Ashwin Kanhere, Derek Knowles"
__date__ = "13 July 2021"

import os
import shutil
import gzip
import ftplib
from ftplib import FTP_TLS, FTP
from datetime import datetime, timezone, timedelta, time

import unlzw3
import numpy as np

import gnss_lib_py.utils.time_conversions as tc

DEFAULT_EPHEM_PATH = os.path.join(os.getcwd(), 'data', 'ephemeris')

def load_ephemeris(file_type, gps_millis,
                   constellations=None, file_paths=None,
                   download_directory=DEFAULT_EPHEM_PATH,
                   verbose=False):
    """
    Verify which ephemeris to download if not in file_paths.

    Parameters
    ----------
    file_type : string
        File type to download either "rinex_nav", "sp3", or "clk".
    gps_millis : float or np.ndarray of floats
        GPS milliseconds for which downloaded ephemeris should be
        obtained.
    constellations : list, set, or array-like
        Constellations for which to download ephemeris, i.e. ["gps"].
    file_paths : list, string or path-like
        Paths to existing ephemeris files if they exist.
    download_directory : string or path-like
        Directory where ephemeris files are downloaded if necessary.
    verbose : bool
        Prints extra debugging statements if true.

    Returns
    -------
    file_paths : list
        Paths to downloaded and/or existing ephemeris files. Only files
        that need to be used are returned. Superfluous path inputs that
        are not needed are not returned

    """

    if isinstance(file_paths, (str, os.PathLike)):
        file_paths = [file_paths]

    existing_paths, needed_files = _verify_ephemeris(file_type,
                                                     gps_millis,
                                                     constellations,
                                                     file_paths,
                                                     verbose)
    try:
        downloaded_paths = _download_ephemeris(file_type,
                                               needed_files,
                                               download_directory, verbose)
    except ftplib.error_perm as err:
        # try second options for some files
        if verbose:
            print(err)
        if "Connection timed out" in str(err): #pragma: no cover
            if verbose:
                print("Retrying download.")
        elif "WUM0MGXFIN" in str(err) and file_type == "clk":
            if verbose:
                print("Retrying download with GFZ0MGXRAP.")
            needed_files = [(x[0],x[1].replace("WUM0MGXFIN","GFZ0MGXRAP"))
                            for x in needed_files]
        elif "wum" in str(err) and file_type == "clk":
            if verbose:
                print("Retrying download with gbm.")
            needed_files = [(x[0],x[1].replace("wum","gbm"))
                            for x in needed_files]
        elif "BRDM00DLR" in str(err) and file_type == "rinex_nav":
            if verbose:
                print("Retrying download with BRDC00IGS.")
            needed_files = [(x[0],x[1].replace("BRDM00DLR_S","BRDC00IGS_R"))
                            for x in needed_files]
            needed_files = [(x[0],x[1].replace("BRDM00DLR_R","BRDC00IGS_R"))
                            for x in needed_files]

    try:
        # second download attempt
        downloaded_paths = _download_ephemeris(file_type,
                                               needed_files,
                                               download_directory,
                                               verbose)
    except ftplib.error_perm as err:
        # on Nov 26, 2013 - Dec 5, 2013 the entire DDD/YYp/ directory
        # is missing, so use gps and glonass only in that case.
        if "BRDC00IGS_R" in str(err) and file_type == "rinex_nav":
            if verbose:
                print("Retrying download with gps/glonass only.")
            _,needed_gps = _verify_ephemeris(file_type,
                                             gps_millis,
                                             ["gps"],
                                             verbose=verbose)
            _,needed_glonass = _verify_ephemeris(file_type,
                                                 gps_millis,
                                                 ["glonass"],
                                                 verbose=verbose)
            needed_files = needed_gps + needed_glonass
        # third download attempt
        downloaded_paths = _download_ephemeris(file_type,
                                               needed_files,
                                               download_directory,
                                               verbose)

    if verbose:
        if len(existing_paths) > 0:
            print("Using the following existing files:")
            for file in existing_paths:
                print(file)

    file_paths = existing_paths + downloaded_paths

    return file_paths

def _verify_ephemeris(file_type, gps_millis, constellations=None,
                      file_paths=None, verbose=False):
    """Check what ephemeris files to download and if they already exist.

    Parameters
    ----------
    file_type : string
        File type to download either "rinex_nav", "sp3", or "clk".
    gps_millis : float or array-like of floats
        GPS milliseconds for which downloaded ephemeris should be
        obtained.
    constellations : list, set, or array-like
        Constellations for which to download ephemeris, i.e. ["gps"].
    file_paths : string or path-like
        Paths to existing ephemeris files if they exist.
    verbose : bool
        Prints extra debugging statements if true.

    Returns
    -------
    existing_paths : list
        List of existing paths to files from input that will be used.
    needed_files : list
        List of files to download for ephemeris.

    """

    dt_timestamps = np.atleast_1d(tc.gps_millis_to_datetime(gps_millis))

    dates_needed = _extract_ephemeris_dates(file_type, dt_timestamps)
    if verbose:
        print("ephemeris dates needed:",dates_needed)

    existing_paths = []
    needed_files = []
    for date in dates_needed:
        possible_types = []

        if file_type == "rinex_nav":
            if date < datetime(2013,1,1).date():
                raise RuntimeError("gnss_lib_py cannot automatically " \
                                 + "download rinex nav files for "\
                                 + "times before Jan 1, 2013")
            if datetime.now(timezone.utc).date() == date:
                possible_types = ["rinex_nav_today"]
            else:
                if constellations is not None and list(constellations) == ["gps"]:
                    possible_types = ["rinex_nav_gps"]
                elif constellations is not None and list(constellations) == ["glonass"]:
                    possible_types = ["rinex_nav_glonass"]

                # download from day's stream if too early in the day
                # that combined file is not yet uploaded to CDDIS.
                if datetime.now(timezone.utc) < datetime.combine(date+timedelta(days=1),
                                                        time(12)).astimezone(timezone.utc): # pragma: no cover
                    possible_types += ["rinex_nav_today"]
                else:
                    if date < datetime(2019, 11, 25).date():
                        possible_types += ["rinex_nav_multi_r"]
                    else:
                        possible_types += ["rinex_nav_multi_s"]

        if file_type == "sp3":
            if date < datetime(2012,5,25).date():
                raise RuntimeError("gnss_lib_py cannot automatically " \
                                 + "download sp3 files for "\
                                 + "times before May 25, 2012")
            if datetime.now(timezone.utc).date() - timedelta(days=3) < date:
                possible_types += ["sp3_rapid_CODE"]
            elif datetime.now(timezone.utc).date() - timedelta(days=14) < date:
                possible_types += ["sp3_rapid_GFZ"]
            elif date >= datetime(2017, 8, 13).date():
                possible_types += ["sp3_final_CODE"]
            else:
                possible_types += ["sp3_short_CODE"]

        if file_type == "clk":
            if date < datetime(2012,10,14).date():
                raise RuntimeError("gnss_lib_py cannot automatically " \
                                 + "download clk files for "\
                                 + "times before Oct 14, 2012")
            if datetime.now(timezone.utc).date() - timedelta(days=3) < date:
                possible_types += ["clk_rapid_CODE"]
            elif datetime.now(timezone.utc).date() - timedelta(days=14) < date:
                possible_types += ["clk_rapid_GFZ"]
            elif date >= datetime(2020, 7, 5).date():
                possible_types += ["clk_final_CODE"]
            elif date >= datetime(2019, 1, 1).date():
                possible_types += ["clk_final_WUM"]
            elif date >= datetime(2018, 12, 30).date():
                possible_types += ["clk_short_GFZ"]
            elif date >= datetime(2017, 8, 13).date():
                possible_types += ["clk_short_WUM"]
            else:
                possible_types += ["clk_short_CODE"]

        already_exists, filepath = _valid_ephemeris_in_paths(date,
                                                possible_types, file_paths)
        if already_exists:
            existing_paths.append(filepath)
        else:
            needed_files.append(filepath)

    return existing_paths, needed_files

def _download_ephemeris(file_type, needed_files, download_directory,
                       verbose=False):
    """Download ephemeris files.

    Parameters
    ----------
    file_type : string
        File type to download either "rinex_nav", "sp3", or "clk".
    needed_files : list
        List of files to download for ephemeris.
    download_directory : string or path-like
        Directory where ephemeris files are downloaded if necessary.
    verbose : bool
        Prints extra debugging statements if true.

    Returns
    -------
    downloaded_paths : string
        Paths to downloaded and/or existing ephemeris files.

    """

    downloaded_paths = []

    directory = os.path.join(download_directory,*file_type.split("_"))
    os.makedirs(directory, exist_ok=True)

    for url, ftp_path in needed_files:
        filename = ".".join(os.path.split(ftp_path)[1].split(".")[:-1])
        dest_filepath = os.path.join(directory,filename)
        if os.path.isfile(dest_filepath):
            downloaded_paths.append(dest_filepath)
            if verbose:
                print("using previously downloaded file:\n",dest_filepath)
            continue
        dest_path_with_extension = os.path.join(directory,os.path.split(ftp_path)[1])
        _ftp_download(url, ftp_path, dest_path_with_extension,
                      verbose)
        downloaded_paths.append(dest_filepath)

    return downloaded_paths

def _extract_ephemeris_dates(file_type, dt_timestamps):
    """Figure out which dates ephemeris is needed for from datetimes.

    Rinex files are only guaranteed to have data for between 02:00 and
    22:00. If the timestamp is bewteen 00:00 and 02:00, the previous day
    will also be included. If after 22:00, the next day will also be
    included.

    To appropriately compute fit satellite position and clock esimates,
    sp3 and clk files are also pulled for the previous or next day if
    the timestamps are close to the start or end of the day.

    Parameters
    ----------
    file_type : string
        File type to download either "rinex_nav", "sp3", or "clk".
    dt_timestamps : np.ndarray
        Datetime timestamps.

    Returns
    -------
    needed_dates : List
        List of datetime.date objects of the days in UTC for which
        ephemeris needs to be retrieved in order sorted by date.

    """

    # convert all timezones to UTC
    dt_timestamps = [tc.tzinfo_to_utc(t) for t in dt_timestamps]

    needed_dates = set()

    if file_type in ("rinex_nav","sp3","clk"):
        # add every day for each timestamp
        needed_dates.update({dt.date() for dt in dt_timestamps})

        # add every day before if timestamp falls between 0am-2am
        needed_dates.update({dt.date() - timedelta(days=1) for dt in dt_timestamps
                        if (dt <= datetime.combine(dt.date(),
                                        time(2,tzinfo=timezone.utc)))
                        })
        # add every day after if timestamp falls between 10pm and midnight
        #   but only if the datetime is not the current day
        needed_dates.update({dt.date() + timedelta(days=1) for dt in dt_timestamps
                        if ((dt >= datetime.combine(dt.date(),
                                        time(22,tzinfo=timezone.utc))) &
                             (dt.date() != datetime.now(timezone.utc).date()))
                        })

    else:
        raise RuntimeError("invalid file_type variable option")

    needed_dates = sorted(list(needed_dates))

    return needed_dates

def _valid_ephemeris_in_paths(date, possible_types, file_paths=None):
    """Check whether a valid ephemeris already exists in file_paths.

    See file header for detailed documentation on the methodology on
    the sources used to downloaded files.

    Parameters
    ----------
    date : datetime.date
        Days in UTC for which ephemeris needs to be retrieved.
    possible_types : list
        What file types would fulfill the requirement in preference
        order.
    file_paths : string or path-like
        Paths to existing ephemeris files if they exist.

    Returns
    -------
    valid : bool
        Whether or not a valid ephemeris already exists. If true, then
        the correct ephemeris file already exists in file_paths.
        If false, a new file will be downloaded unless file already
        exists in the download directory.
    recommended_file : string
        Path to existing file if valid is True, otherwise a tuple
        containing the url and filepath to the file that should be
        downloaded.

    """

    timetuple = date.timetuple()
    recommended_files = []

    for possible_type in possible_types:

        # Rinex for the current day
        if possible_type == "rinex_nav_today":
            recommended_file = ("igs-ftp.bkg.bund.de",
                                "/IGS/BRDC/" + str(timetuple.tm_year) \
                              + "/" + str(timetuple.tm_yday).zfill(3) \
                              + "/" + "BRDC00WRD_S_" \
                              + str(timetuple.tm_year) \
                              + str(timetuple.tm_yday).zfill(3) \
                              + "0000_01D_MN.rnx.gz")
            recommended_files.append(recommended_file)
            if file_paths is None:
                return False, recommended_file
            # check compatible file types
            for path in file_paths:
                if os.path.split(path)[1] + ".gz" == os.path.split(recommended_file[1])[1]:
                    return True, path

        # rinex for multi-gnss if before Nov 25, 2019
        elif possible_type == "rinex_nav_multi_r":
            recommended_file = ("gdc.cddis.eosdis.nasa.gov",
                                "/gnss/data/daily/" \
                              + str(timetuple.tm_year) + "/" \
                              + str(timetuple.tm_yday).zfill(3) + "/" \
                              + str(timetuple.tm_year)[-2:] + 'p' + "/"\
                              + "BRDM00DLR_R_" + str(timetuple.tm_year)\
                              + str(timetuple.tm_yday).zfill(3) \
                              + "0000_01D_MN.rnx.gz")
            recommended_files.append(recommended_file)
            if file_paths is None:
                return False, recommended_file
            # check compatible file types
            for path in file_paths:
                if os.path.split(path)[1] + ".gz" == os.path.split(recommended_file[1])[1]:
                    return True, path
            for path in file_paths:
                if os.path.split(path)[1][-22:] == recommended_file[1][-25:-3]:
                    return True, path

        # rinex for multi-gnss on or after Nov 25, 2019
        elif possible_type == "rinex_nav_multi_s":
            recommended_file = ("gdc.cddis.eosdis.nasa.gov",
                                "/gnss/data/daily/" \
                              + str(timetuple.tm_year) + "/" \
                              + str(timetuple.tm_yday).zfill(3) + "/" \
                              + str(timetuple.tm_year)[-2:] + 'p' + "/"\
                              + "BRDM00DLR_S_" + str(timetuple.tm_year)\
                              + str(timetuple.tm_yday).zfill(3) \
                              + "0000_01D_MN.rnx.gz")
            recommended_files.append(recommended_file)
            if file_paths is None:
                return False, recommended_file
            # check compatible file types
            for path in file_paths:
                if os.path.split(path)[1] + ".gz" == os.path.split(recommended_file[1])[1]:
                    return True, path
            for path in file_paths:
                if os.path.split(path)[1][-22:] == recommended_file[1][-25:-3]:
                    return True, path

        # rinex that only contains GPS
        elif possible_type == "rinex_nav_gps":
            recommended_file = ("gdc.cddis.eosdis.nasa.gov",
                                "/gnss/data/daily/" \
                              + str(timetuple.tm_year) + "/" \
                              + str(timetuple.tm_yday).zfill(3) + "/" \
                              + str(timetuple.tm_year)[-2:] + 'n' + "/"\
                              + "brdc"+ str(timetuple.tm_yday).zfill(3)\
                              + "0." + str(timetuple.tm_year)[-2:] +'n'\
                              + _get_rinex_extension(date))
            recommended_files.append(recommended_file)
            if file_paths is None:
                return False, recommended_file
            # check compatible file types
            for path in file_paths:
                if os.path.split(path)[1] + _get_rinex_extension(date) == os.path.split(recommended_file[1])[1]:
                    return True, path
            for path in file_paths:
                if os.path.split(path)[1][4:] == str(timetuple.tm_yday).zfill(3)\
                                               + "0." + str(timetuple.tm_year)[-2:]\
                                               +'n':
                    return True, path
            long_name = str(timetuple.tm_year)\
                      + str(timetuple.tm_yday).zfill(3) \
                      + "0000_01D_GN.rnx"
            for path in file_paths:
                if os.path.split(path)[1][-22:] == long_name:
                    return True, path

        # rinex that only contains GLONASS
        elif possible_type == "rinex_nav_glonass":
            recommended_file = ("gdc.cddis.eosdis.nasa.gov",
                                "/gnss/data/daily/" \
                              + str(timetuple.tm_year) + "/" \
                              + str(timetuple.tm_yday).zfill(3) + "/" \
                              + str(timetuple.tm_year)[-2:] + 'g' + "/"\
                              + "brdc"+ str(timetuple.tm_yday).zfill(3)\
                              + "0." + str(timetuple.tm_year)[-2:] +'g'\
                              + _get_rinex_extension(date))
            recommended_files.append(recommended_file)
            if file_paths is None:
                return False, recommended_file
            # check compatible file types
            for path in file_paths:
                if os.path.split(path)[1] + _get_rinex_extension(date) == os.path.split(recommended_file[1])[1]:
                    return True, path
            for path in file_paths:
                if os.path.split(path)[1][4:] == str(timetuple.tm_yday).zfill(3)\
                                               + "0." + str(timetuple.tm_year)[-2:]\
                                               +'g':
                    return True, path
            long_name = str(timetuple.tm_year)\
                      + str(timetuple.tm_yday).zfill(3) \
                      + "0000_01D_RN.rnx"
            for path in file_paths:
                if os.path.split(path)[1][-22:] == long_name:
                    return True, path

        # sp3 from last three days
        elif possible_type == "sp3_rapid_CODE":
            gps_week, _ = tc.datetime_to_tow(datetime.combine(date,
                                         time(tzinfo=timezone.utc)))
            recommended_file = ("gdc.cddis.eosdis.nasa.gov",
                                "/gnss/products/" \
                              + str(gps_week).zfill(4) + "/" \
                              + "COD0OPSRAP_" + str(timetuple.tm_year) \
                              + str(timetuple.tm_yday).zfill(3) \
                              + "0000_01D_05M_ORB.SP3.gz")
            recommended_files.append(recommended_file)
            if file_paths is None:
                return False, recommended_file
            # check compatible file types
            for path in file_paths:
                if os.path.split(path)[1] + ".gz" == os.path.split(recommended_file[1])[1]:
                    return True, path
            for path in file_paths:
                if os.path.split(path)[1][10:] == os.path.split(recommended_file[1])[1][10:-3]:
                    return True, path
            for path in file_paths:
                if os.path.split(path)[1][3:] == str(gps_week).zfill(4) + str((timetuple.tm_wday+1)%7) + ".sp3":
                    return True, path

        # sp3 from last two weeks
        elif possible_type == "sp3_rapid_GFZ":
            gps_week, _ = tc.datetime_to_tow(datetime.combine(date,
                                         time(tzinfo=timezone.utc)))
            recommended_file = ("gdc.cddis.eosdis.nasa.gov",
                                "/gnss/products/" \
                              + str(gps_week).zfill(4) + "/" \
                              + "GFZ0MGXRAP_" + str(timetuple.tm_year) \
                              + str(timetuple.tm_yday).zfill(3) \
                              + "0000_01D_05M_ORB.SP3.gz")
            recommended_files.append(recommended_file)
            if file_paths is None:
                return False, recommended_file
            # check compatible file types
            for path in file_paths:
                if os.path.split(path)[1] + ".gz" == os.path.split(recommended_file[1])[1]:
                    return True, path
            for path in file_paths:
                if os.path.split(path)[1][10:] == os.path.split(recommended_file[1])[1][10:-3]:
                    return True, path
            for path in file_paths:
                if os.path.split(path)[1][3:] == str(gps_week).zfill(4) + str((timetuple.tm_wday+1)%7) + ".sp3":
                    return True, path

        # sp3 if longer than two weeks ago and more recent than Aug 13, 2017
        elif possible_type == "sp3_final_CODE":
            gps_week, _ = tc.datetime_to_tow(datetime.combine(date,
                                         time(tzinfo=timezone.utc)))
            recommended_file = ("gdc.cddis.eosdis.nasa.gov",
                                "/gnss/products/" \
                              + str(gps_week).zfill(4) + "/" \
                              + "COD0MGXFIN_" + str(timetuple.tm_year) \
                              + str(timetuple.tm_yday).zfill(3) \
                              + "0000_01D_05M_ORB.SP3.gz")
            recommended_files.append(recommended_file)
            if file_paths is None:
                return False, recommended_file
            # check compatible file types
            for path in file_paths:
                if os.path.split(path)[1] + ".gz" == os.path.split(recommended_file[1])[1]:
                    return True, path
            for path in file_paths:
                if os.path.split(path)[1][10:] == os.path.split(recommended_file[1])[1][10:-3]:
                    return True, path
            for path in file_paths:
                if os.path.split(path)[1][3:] == str(gps_week).zfill(4) + str((timetuple.tm_wday+1)%7) + ".sp3":
                    return True, path

        # sp3 between May 25, 2012 and Aug 12, 2017 inclusive
        elif possible_type == "sp3_short_CODE":
            gps_week, _ = tc.datetime_to_tow(datetime.combine(date,
                                         time(tzinfo=timezone.utc)))
            recommended_file = ("gdc.cddis.eosdis.nasa.gov",
                                "/gnss/products/" \
                              + str(gps_week).zfill(4) + "/" \
                              + "com" + str(gps_week).zfill(4) \
                              + str((timetuple.tm_wday+1)%7) \
                              + ".sp3.Z")
            recommended_files.append(recommended_file)
            if file_paths is None:
                return False, recommended_file
            # check compatible file types
            for path in file_paths: # exact match
                if os.path.split(path)[1] + ".Z" == os.path.split(recommended_file[1])[1]:
                    return True, path
            for path in file_paths: # other short form files
                if os.path.split(path)[1][3:] + ".Z" == os.path.split(recommended_file[1])[1][3:]:
                    return True, path
            for path in file_paths: # other long form files
                if os.path.split(path)[1][11:] == str(timetuple.tm_year) \
                                                + str(timetuple.tm_yday).zfill(3) \
                                                + "0000_01D_05M_ORB.SP3":
                    return True, path

        # clk from last three days
        elif possible_type == "clk_rapid_CODE":
            gps_week, _ = tc.datetime_to_tow(datetime.combine(date,
                                         time(tzinfo=timezone.utc)))
            recommended_file = ("gdc.cddis.eosdis.nasa.gov",
                                "/gnss/products/" \
                              + str(gps_week).zfill(4) + "/" \
                              + "COD0OPSRAP_" + str(timetuple.tm_year) \
                              + str(timetuple.tm_yday).zfill(3) \
                              + "0000_01D_30S_CLK.CLK.gz")
            recommended_files.append(recommended_file)
            if file_paths is None:
                return False, recommended_file
            # check compatible file types
            for path in file_paths:
                if os.path.split(path)[1] + ".gz" == os.path.split(recommended_file[1])[1]:
                    return True, path
            for path in file_paths:
                if os.path.split(path)[1][10:] == os.path.split(recommended_file[1])[1][10:-3]:
                    return True, path
            for path in file_paths:
                if os.path.split(path)[1][3:] == str(gps_week).zfill(4) + str((timetuple.tm_wday+1)%7) + ".clk":
                    return True, path

        # clk from last two weeks
        elif possible_type == "clk_rapid_GFZ":
            gps_week, _ = tc.datetime_to_tow(datetime.combine(date,
                                         time(tzinfo=timezone.utc)))
            recommended_file = ("gdc.cddis.eosdis.nasa.gov",
                                "/gnss/products/" \
                              + str(gps_week).zfill(4) + "/" \
                              + "GFZ0MGXRAP_" + str(timetuple.tm_year) \
                              + str(timetuple.tm_yday).zfill(3) \
                              + "0000_01D_30S_CLK.CLK.gz")
            recommended_files.append(recommended_file)
            if file_paths is None:
                return False, recommended_file
            # check compatible file types
            for path in file_paths:
                if os.path.split(path)[1] + ".gz" == os.path.split(recommended_file[1])[1]:
                    return True, path
            for path in file_paths:
                if os.path.split(path)[1][10:] == os.path.split(recommended_file[1])[1][10:-3]:
                    return True, path
            for path in file_paths:
                if os.path.split(path)[1][3:] == str(gps_week).zfill(4) + str((timetuple.tm_wday+1)%7) + ".clk":
                    return True, path

        # clk if longer than two weeks ago and more recent than Jul 5, 2020
        elif possible_type == "clk_final_CODE":
            gps_week, _ = tc.datetime_to_tow(datetime.combine(date,
                                         time(tzinfo=timezone.utc)))
            recommended_file = ("gdc.cddis.eosdis.nasa.gov",
                                "/gnss/products/" \
                              + str(gps_week).zfill(4) + "/" \
                              + "COD0MGXFIN_" + str(timetuple.tm_year) \
                              + str(timetuple.tm_yday).zfill(3) \
                              + "0000_01D_30S_CLK.CLK.gz")
            recommended_files.append(recommended_file)
            if file_paths is None:
                return False, recommended_file
            # check compatible file types
            for path in file_paths:
                if os.path.split(path)[1] + ".gz" == os.path.split(recommended_file[1])[1]:
                    return True, path
            for path in file_paths:
                if os.path.split(path)[1][10:] == os.path.split(recommended_file[1])[1][10:-3]:
                    return True, path
            for path in file_paths:
                if os.path.split(path)[1][3:] == str(gps_week).zfill(4) + str((timetuple.tm_wday+1)%7) + ".clk":
                    return True, path

        # clk for Jan 1, 2019 to Jul 4, 2020
        elif possible_type == "clk_final_WUM":
            gps_week, _ = tc.datetime_to_tow(datetime.combine(date,
                                         time(tzinfo=timezone.utc)))
            recommended_file = ("gdc.cddis.eosdis.nasa.gov",
                                "/gnss/products/" \
                              + str(gps_week).zfill(4) + "/" \
                              + "WUM0MGXFIN_" + str(timetuple.tm_year) \
                              + str(timetuple.tm_yday).zfill(3) \
                              + "0000_01D_30S_CLK.CLK.gz")
            recommended_files.append(recommended_file)
            if file_paths is None:
                return False, recommended_file
            # check compatible file types
            for path in file_paths:
                if os.path.split(path)[1] + ".gz" == os.path.split(recommended_file[1])[1]:
                    return True, path
            for path in file_paths:
                if os.path.split(path)[1][10:] == os.path.split(recommended_file[1])[1][10:-3]:
                    return True, path
            for path in file_paths:
                if os.path.split(path)[1][3:] == str(gps_week).zfill(4) + str((timetuple.tm_wday+1)%7) + ".clk":
                    return True, path

        # clk for Dec 30, 2018 to Dec 31, 2018
        elif possible_type == "clk_short_GFZ":
            gps_week, _ = tc.datetime_to_tow(datetime.combine(date,
                                         time(tzinfo=timezone.utc)))
            recommended_file = ("gdc.cddis.eosdis.nasa.gov",
                                "/gnss/products/" \
                              + str(gps_week).zfill(4) + "/" \
                              + "gbm" + str(gps_week).zfill(4) \
                              + str((timetuple.tm_wday+1)%7) \
                              + ".clk.Z")
            recommended_files.append(recommended_file)
            if file_paths is None:
                return False, recommended_file
            # check compatible file types
            for path in file_paths: # exact file match
                if os.path.split(path)[1] + ".Z" == os.path.split(recommended_file[1])[1]:
                    return True, path
            for path in file_paths: # other short form files
                if os.path.split(path)[1][3:] + ".Z" == os.path.split(recommended_file[1])[1][3:]:
                    return True, path
            for path in file_paths: # other long form files
                if os.path.split(path)[1][11:] == str(timetuple.tm_year) \
                                                + str(timetuple.tm_yday).zfill(3) \
                                                + "0000_01D_30S_CLK.CLK":
                    return True, path

        # clk for Aug 13, 2017 to Dec 29, 2018
        elif possible_type == "clk_short_WUM":
            gps_week, _ = tc.datetime_to_tow(datetime.combine(date,
                                         time(tzinfo=timezone.utc)))
            recommended_file = ("gdc.cddis.eosdis.nasa.gov",
                                "/gnss/products/" \
                              + str(gps_week).zfill(4) + "/" \
                              + "wum" + str(gps_week).zfill(4) \
                              + str((timetuple.tm_wday+1)%7) \
                              + ".clk.Z")
            recommended_files.append(recommended_file)
            if file_paths is None:
                return False, recommended_file
            # check compatible file types
            for path in file_paths:
                if os.path.split(path)[1] + ".Z" == os.path.split(recommended_file[1])[1]:
                    return True, path
            for path in file_paths:
                if os.path.split(path)[1][3:] + ".Z" == os.path.split(recommended_file[1])[1][3:]:
                    return True, path
            for path in file_paths: # other long form files
                if os.path.split(path)[1][11:] == str(timetuple.tm_year) \
                                                + str(timetuple.tm_yday).zfill(3) \
                                                + "0000_01D_30S_CLK.CLK":
                    return True, path

        # clk for Oct 14, 2012 to Aug 12, 2017
        elif possible_type == "clk_short_CODE":
            gps_week, _ = tc.datetime_to_tow(datetime.combine(date,
                                         time(tzinfo=timezone.utc)))
            recommended_file = ("gdc.cddis.eosdis.nasa.gov",
                                "/gnss/products/" \
                              + str(gps_week).zfill(4) + "/" \
                              + "com" + str(gps_week).zfill(4) \
                              + str((timetuple.tm_wday+1)%7) \
                              + ".clk.Z")
            recommended_files.append(recommended_file)
            if file_paths is None:
                return False, recommended_file
            # check compatible file types
            for path in file_paths:
                if os.path.split(path)[1] + ".Z" == os.path.split(recommended_file[1])[1]:
                    return True, path
            for path in file_paths:
                if os.path.split(path)[1][3:] + ".Z" == os.path.split(recommended_file[1])[1][3:]:
                    return True, path
            for path in file_paths: # other long form files
                if os.path.split(path)[1][11:] == str(timetuple.tm_year) \
                                                + str(timetuple.tm_yday).zfill(3) \
                                                + "0000_01D_30S_CLK.CLK":
                    return True, path
        else:
            raise RuntimeError(possible_type,"invalid possible_type "\
                                +"for valid ephemeris")

    return False, recommended_files[0]

FTP_DOWNLOAD_SOURCECODE = "FTP_DOWNLOAD_SOURCECODE"
"""Private FTP functions were pulled from class code.

The functions ``_ftp_download``, ``_ftp_login``, ``_decompress_file``,
and ``_get_rinex_extension`` were modified from code at:
https://github.com/johnsonmitchelld/gnss-analysis/blob/main/gnssutils/ephemeris_manager.py

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

def _ftp_download(url, ftp_path, dest_filepath, verbose=False):
    """Copy ephemeris file from FTP filepath to local directory.

    Also decompresses file.

    Parameters
    ----------
    url : string
        FTP server url.
    ftp_path : string
        Path to ephemeris file stored on the FTP server.
    dest_filepath : string
        File path to downloaded ephemeris file are stored locally.
    verbose : bool
        Prints extra debugging statements if true.

    """

    secure = bool(url == 'gdc.cddis.eosdis.nasa.gov')

    if verbose:
        print('FTP downloading ' + ftp_path + ' from ' + url)
    ftp = _ftp_login(url, secure)
    src_filepath = ftp_path
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
    _decompress_file(dest_filepath)

def _ftp_login(url, secure=False):
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

def _decompress_file(filepath, remove_compressed=True):
    """Decompress ephemeris file in same destination.

    Parameters
    ----------
    filepath : string
        Local filepath where the compressed ephemeris file is stored
        and subsequently decompressed.
    remove_compressed : bool
        If true, will delete the compressed file.

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
    if remove_compressed:
        os.remove(filepath)

def _get_rinex_extension(timestamp):
    """Get file extension of rinex file based on timestamp.

    GPS and Glonass Rinex files switched from .Z to .gz on
    December 1, 2020 [15]_.

    But brdc actually switched on November 30, 2020.

    Parameters
    ----------
    timestamp : datetime.date
        Date of ephemeris file.

    Returns
    -------
    extension : string
        Extension of compressed ephemeris file.

    References
    ----------
    .. [15] https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/daily_30second_data.html

    """

    if timestamp >= datetime(2020, 11, 30, tzinfo=timezone.utc).date():
        extension = '.gz'
    else:
        extension = '.Z'
    return extension
