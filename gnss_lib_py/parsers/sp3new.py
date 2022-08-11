# ECE456_fileutils.py
#
# Contains various functions to parse and work with data files used in the
# ECE/AE 456 course at UIUC.
#
# HISTORY:
#   1) 07-15-2015: Created by J. Makela
#   2) 07-15-2015: J. Makela added parse_pos to handle the pos files generated
#                   by rnx2rtkp.
#   3) 07-15-2015: J. Makela added class and function to read in YUMA alm files.
#   4) 08-13-2015: J. Makela added parse_ubx_lla to retreive LLA data from a ubx file
#   5) 08-27-2015: Yuting Ng changed field names in parse_ubx_lla.
#   6) 09-29-2015: Yuting Ng added parse_ubx_test to check for received messages.

import datetime as datetime
from scipy import interpolate
import numpy as np
import scipy.io
import itertools
import pytz
import gmplot
import csv
#import pandas as pd

# Define the number of sats to create arrays for
NUMSATS = 32
NUMSATS_BEIDOU = 46
NUMSATS_GLONASS = 24
NUMSATS_GALILEO = 36
NUMSATS_QZSS = 3

def utc2gps(dt, leapSeconds = 0):
    """
    Function: utc2gps(dt, leapSeconds)
    ---------------------
    Return the gpsTime and gpsWeek based on the requested time. Based, in part
    on https://www.lsc-group.phys.uwm.edu/daswg/projects/glue/epydoc/lib/python2.4/site-packages/glue/gpstime.py

    Inputs:
    -------
        dt : a datetime (in UTC) to be converted
        leapSeconds : (optional; default = 17) correction for GPS leap seconds
        
    Outputs:
    --------
        gpsTime : the converted GPS time of week [sec] 
        gpsWeek - the GPS week number (without considering rollovers)

    Notes:
    ------
        Based from Jonathan Makela's GPS_GMT2GPS_week.m script

    History:
    --------
        7/15/15 Created, Jonathan Makela (jmakela@illinois.edu)
        
    ToDo:
    --------
        1) Make leapSeconds calculate automatically based on a historical table and the requested date

    """
    
    # Define the GPS epoch
    gpsEpoch = datetime.datetime(1980,1,6,0,0,0)
    gpsEpoch = gpsEpoch.replace(tzinfo = pytz.utc)
    
    # Check if requested time is timezone aware.  If not, assume it is a UT time
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo = pytz.utc)
    
    # Calculate the time delta from epoch
    delta_t = dt - gpsEpoch

    # The total number of seconds (add in GPS leap seconds)
    secsEpoch = delta_t.total_seconds()+leapSeconds
    
    # The gpsTime is the total seconds since epoch mod the number of seconds in a week
    secsInWeek = 604800.
    gpsTime = np.mod(secsEpoch, secsInWeek)
    gpsWeek = int(np.floor(secsEpoch/secsInWeek))

    return gpsTime, gpsWeek


class sp3:
    def __init__(self):
        self.x = []
        self.y = []
        self.z = []
        self.t = []
        self.UTCtime = []    

def extract_sp3(sp3data, tym=-1, const='G', ipos=10, method='CubicSpline'):
    if(const=='G'):
        nSVs = NUMSATS
    elif (const=='C'):
        nSVs = NUMSATS_BEIDOU
    elif (const=='R'):
        nSVs = NUMSATS_GLONASS
    elif (const=='E'):
        nSVs = NUMSATS_GALILEO
    elif (const=='J'):
        nSVs = NUMSATS_QZSS
    totalSATS = np.arange(1,nSVs+1)

    if (tym==-1):
        low_i, high_i = 0, -1
    
    func_satXYZ = np.empty((nSVs+1,3), dtype=object)
    func_satXYZ[:] = np.nan
    for _, prn in enumerate(totalSATS): 
        if sp3data[prn].t:
            if (tym!=-1): 
                sidx = np.argmin(abs(np.array(sp3data[prn].t)-tym))
                low_i = (sidx - ipos) if (sidx - ipos) >= 0 else 0
                high_i = (sidx + ipos) if (sidx + ipos) <= len(sp3data[prn].t) else -1
            if method=='CubicSpline':
                func_satXYZ[prn][0] = interpolate.CubicSpline(sp3data[prn].t[low_i:high_i], \
                                                                sp3data[prn].x[low_i:high_i])
                func_satXYZ[prn][1] = interpolate.CubicSpline(sp3data[prn].t[low_i:high_i], \
                                                                sp3data[prn].y[low_i:high_i])
                func_satXYZ[prn][2] = interpolate.CubicSpline(sp3data[prn].t[low_i:high_i], \
                                                                sp3data[prn].z[low_i:high_i])
            elif method=='polyfit': 
                poly_degree = 14
                func_satXYZ[prn][0] = np.polyfit(sp3data[prn].t, \
                                                 sp3data[prn].x, deg=poly_degree)
                func_satXYZ[prn][1] = np.polyfit(sp3data[prn].t, \
                                                 sp3data[prn].y, deg=poly_degree)
                func_satXYZ[prn][2] = np.polyfit(sp3data[prn].t, \
                                                 sp3data[prn].z, deg=poly_degree) 
            else: 
                print('This interpolation type is not possible!')
                
                
    return func_satXYZ
    
def parse_sp3(fname, const='G'):
    # Parse a .sp3 file for satellite location data.
    # Format definition at https://igscb.jpl.nasa.gov/igscb/data/format/sp3_docu.txt
    #
    # NOTE: I've been lazy and have ingored header information and some of the data columns.  The 
    # current code only parses for the satellite location data.
    #
    # INPUT: fname - full path to the .sp3 file to be parsed
    # HISTORY: Written by J. Makela on 9-2-2015

    # Load in the file
    data = [line.strip() for line in open(fname)]

    # Create a sp3 class for each expected satellite
    sp3data = []
    if(const=='G'):
        nSVs = NUMSATS
    elif (const=='C'):
        nSVs = NUMSATS_BEIDOU
    elif (const=='R'):
        nSVs = NUMSATS_GLONASS
    elif (const=='E'):
        nSVs = NUMSATS_GALILEO
    elif (const=='J'):
        nSVs = NUMSATS_QZSS

    for i in np.arange(0, nSVs+1):
        sp3data.append(sp3())

    # Loop through each line
    for d in data:
        if len(d) == 0:
            # No data
            continue
        if d[0] == '*':
            # A new record
            # Get the date
            temp = d.split()
            myTime = datetime.datetime(int(temp[1]),int(temp[2]),int(temp[3]),int(temp[4]),int(temp[5]),int(float(temp[6])))
            GPStym, _ = utc2gps(myTime)

        if 'P' in d[0]:
            # A satellite record.  Get the satellite number, and coordinate (X,Y,Z) info
            temp = d.split()
            if (temp[0][1]==const):
                #print(temp[0][1])
                prn = int(temp[0][2:])
                #print(prn)
                sp3data[prn].UTCtime.append(myTime)
                sp3data[prn].t.append(GPStym)
                sp3data[prn].x.append(float(temp[1])*1e3)
                sp3data[prn].y.append(float(temp[2])*1e3)
                sp3data[prn].z.append(float(temp[3])*1e3)

    return sp3data

class SVclk:
    def __init__(self):
        self.CLKbias = []
        self.UTCtime = []   
        self.t = [] 

def parse_clockFile(clkFile, const='G'):

    # Create a CLK class for each expected satellite
    clkdata = []
    if(const=='G'):
        nSVs = NUMSATS
    elif (const=='C'):
        nSVs = NUMSATS_BEIDOU
    elif (const=='R'):
        nSVs = NUMSATS_GLONASS
    elif (const=='E'):
        nSVs = NUMSATS_GALILEO
    elif (const=='J'):
        nSVs = NUMSATS_QZSS

    for i in np.arange(0, nSVs+1):
        clkdata.append(SVclk())

    """ Read Clock file """
    f = open(clkFile)
    clk = f.readlines()
    line = 0

    while True:
        if 'OF SOLN SATS' not in clk[line]:
            del clk[line]
        else:   
            noprn = int(clk[line][4:6])
            line +=1
            break
    
    line = 0
    while True:
        if 'END OF HEADER' not in clk[line]:
            line +=1
        else: 
            del clk[0:line+1]
            break

    timelist = []
    for i in range(len(clk)):
        if clk[i][0:2]=='AS':
            timelist.append(clk[i].split())

    for i in range(len(timelist)):
        d = timelist[i][1]
        if (d[0]==const):
            prn = int(d[1:])
            myTime = datetime.datetime(year = int(timelist[i][2]),  month = int(timelist[i][3]),
                                       day = int(timelist[i][4]),   hour = int(timelist[i][5]), 
                                       minute = int(timelist[i][6]), second = int(float(timelist[i][7])))
            GPStym, _ = utc2gps(myTime)
            clkdata[prn].UTCtime.append(myTime)
            clkdata[prn].t.append(GPStym)
            clkdata[prn].CLKbias.append(float(timelist[i][9]))
    f.close() # close the file

    return clkdata

def extract_clk(clkdata, tym=-1, const='G', ipos=10):
    if(const=='G'):
        nSVs = NUMSATS
    elif (const=='C'):
        nSVs = NUMSATS_BEIDOU
    elif (const=='R'):
        nSVs = NUMSATS_GLONASS
    elif (const=='E'):
        nSVs = NUMSATS_GALILEO
    elif (const=='J'):
        nSVs = NUMSATS_QZSS
    totalSATS = np.arange(1,nSVs+1)

    if (tym==-1):
        low_i, high_i = 0, -1
    
    func_clkBIAS = np.empty((nSVs+1,1), dtype=object)
    func_clkBIAS[:] = np.nan
    for _, prn in enumerate(totalSATS): 
        if clkdata[prn].t:
            if (tym!=-1): 
                sidx = np.argmin(abs(np.array(clkdata[prn].t)-tym))
                low_i = (sidx - ipos) if (sidx - ipos) >= 0 else 0
                high_i = (sidx + ipos) if (sidx + ipos) <= len(clkdata[prn].t) else -1
                
            func_clkBIAS[prn] = interpolate.CubicSpline(clkdata[prn].t[low_i:high_i], \
                                                        clkdata[prn].CLKbias[low_i:high_i])
    return func_clkBIAS

