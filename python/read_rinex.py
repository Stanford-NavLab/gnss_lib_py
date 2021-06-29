from datetime import datetime, timedelta
from io import BytesIO
import pandas as pd
import numpy as np
from .constants import *

#Read RINEX file
def _obstime(fol):
    """
    Python >= 3.7 supports nanoseconds.  https://www.python.org/dev/peps/pep-0564/
    Python < 3.7 supports microseconds.
    """
    year = int(fol[0])
    if 80 <= year <= 99:
        year += 1900
    elif year < 80:  # because we might pass in four-digit year
        year += 2000

    return datetime(year=year, month=int(fol[1]), day=int(fol[2]),
                    hour=int(fol[3]), minute=int(fol[4]),
                    second=int(float(fol[5])),
                    microsecond=int(float(fol[5]) % 1 * 1000000)
                    )

def read_rinex2(input_path):
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
      fields = ['SVclockBias', 'SVclockDrift', 'SVclockDriftRate', 'IODE', 'Crs', 'DeltaN',
                'M0', 'Cuc', 'Eccentricity', 'Cus', 'sqrtA', 'Toe', 'Cic', 'Omega0', 'Cis', 'Io',
                'Crc', 'omega', 'OmegaDot', 'IDOT', 'CodesL2', 'GPSWeek', 'L2Pflag', 'SVacc',
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
      dt.append(_obstime([ln[3:5], ln[6:8], ln[9:11],
                          ln[12:14], ln[15:17], ln[17:20], ln[17:22]]))
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

#Ephemeris to ECEF coordinates 
def _kepler(mk, e):
  ek = mk
  count = 0
  max_count = 20
  err = 1.
  while (np.abs(err)>1e-8) and (count < max_count):
    err = ek - mk - e*np.sin(ek)
    ek = ek - err
    count += 1
  return ek

  #TODO: Delete versions of the code 

def _ephem2xyz(ephem, data):
  tk = (data.Week-ephem.GPSWeek)*604800 + (data.ReceivedSvTimeNanos/1e9 - ephem.Toe)      # Time since time of applicability accounting for weeks
  A = ephem.sqrtA**2      # semi-major axis of orbit
  n0 = np.sqrt(3.986005e14/A**3)   # Computed mean motion (rad/sec)
  n = n0 + ephem.DeltaN     # Corrected Mean Motion
  Mk = ephem.M0 + n*tk   # Mean Anomaly
  Ek = _kepler(Mk, ephem.Eccentricity)    # Solve Kepler's equation for eccentric anomaly
  dt = (data.Week-ephem.GPSWeek)*604800 + (data.ReceivedSvTimeNanos/1e9 - ephem.TransTime)    # Calculate satellite clock bias (See ICD-GPS-200 20.3.3.3.3.1)
  sin_Ek = np.sin(Ek)
  cos_Ek = np.cos(Ek)
  dtsvS = ephem.SVclockBias + ephem.SVclockDrift*dt + ephem.SVclockDriftRate*(dt**2) + -4.442807633e-10*ephem.Eccentricity*ephem.sqrtA*sin_Ek - ephem.TGD
  vk = np.arctan2(np.sqrt(1-ephem.Eccentricity**2)*sin_Ek/(1-ephem.Eccentricity*cos_Ek) , (cos_Ek-ephem.Eccentricity)/(1-ephem.Eccentricity*cos_Ek))
  Phik = vk + ephem.omega

  sin_2Phik = np.sin(2*Phik)
  cos_2Phik = np.cos(2*Phik)

  duk = ephem.Cus*sin_2Phik + ephem.Cuc*cos_2Phik   # Argument of latitude correction
  drk = ephem.Crc*cos_2Phik + ephem.Crs*sin_2Phik   # Radius Correction
  dik = ephem.Cic*cos_2Phik + ephem.Cis*sin_2Phik   # Correction to Inclination
  
  uk = Phik + duk
  rk = A*((1-ephem.Eccentricity**2)/(1+ephem.Eccentricity*np.cos(vk))) + drk
  ik = ephem.Io + ephem.IDOT*tk + dik

  sin_uk = np.sin(uk)
  cos_uk = np.cos(uk)
  xkp = rk*cos_uk     # Position in orbital plane
  ykp = rk*sin_uk     # Position in orbital plane

  Wk  = ephem.Omega0 + (ephem.OmegaDot - 7.2921151467e-5)*tk - 7.2921151467e-5*ephem.Toe

  sin_Wk = np.sin(Wk)
  cos_Wk = np.cos(Wk)

  sin_ik = np.sin(ik)
  cos_ik = np.cos(ik)

  X = xkp*cos_Wk - ykp*cos_ik*sin_Wk
  Y = xkp*sin_Wk + ykp*cos_ik*cos_Wk
  Z = ykp*sin_ik
  return X, Y, Z

# satellite position from transmit time
def calc_satpos(ephemeris, transmit_time):
  mu = 3.986005e14
  OmegaDot_e = 7.2921151467e-5
  F = -4.442807633e-10
  sv_position = pd.DataFrame()
  sv_position['sv']= ephemeris.index
  sv_position.set_index('sv', inplace=True)
  sv_position['t_k'] = transmit_time - ephemeris['t_oe']
  A = ephemeris['sqrtA'].pow(2)
  n_0 = np.sqrt(mu / A.pow(3))
  n = n_0 + ephemeris['deltaN']
  M_k = ephemeris['M_0'] + n * sv_position['t_k']
  E_k = M_k
  err = pd.Series(data=[1]*len(sv_position.index))
  i = 0
  while err.abs().min() > 1e-8 and i < 10:
      new_vals = M_k + ephemeris['e']*np.sin(E_k)
      err = new_vals - E_k
      E_k = new_vals
      i += 1
      
  sinE_k = np.sin(E_k)
  cosE_k = np.cos(E_k)
  delT_r = F * ephemeris['e'].pow(ephemeris['sqrtA']) * sinE_k
  delT_oc = transmit_time - ephemeris['t_oc']
  sv_position['delT_sv'] = ephemeris['SVclockBias'] + ephemeris['SVclockDrift'] * delT_oc + ephemeris['SVclockDriftRate'] * delT_oc.pow(2)

  v_k = np.arctan2(np.sqrt(1-ephemeris['e'].pow(2))*sinE_k,(cosE_k - ephemeris['e']))

  Phi_k = v_k + ephemeris['omega']

  sin2Phi_k = np.sin(2*Phi_k)
  cos2Phi_k = np.cos(2*Phi_k)

  du_k = ephemeris['C_us']*sin2Phi_k + ephemeris['C_uc']*cos2Phi_k
  dr_k = ephemeris['C_rs']*sin2Phi_k + ephemeris['C_rc']*cos2Phi_k
  di_k = ephemeris['C_is']*sin2Phi_k + ephemeris['C_ic']*cos2Phi_k

  u_k = Phi_k + du_k

  r_k = A*(1 - ephemeris['e']*np.cos(E_k)) + dr_k

  i_k = ephemeris['i_0'] + di_k + ephemeris['IDOT']*sv_position['t_k']

  x_k_prime = r_k*np.cos(u_k)
  y_k_prime = r_k*np.sin(u_k)

  Omega_k = ephemeris['Omega_0'] + (ephemeris['OmegaDot'] - OmegaDot_e)*sv_position['t_k'] - OmegaDot_e*ephemeris['t_oe']

  sv_position['x_k'] = x_k_prime*np.cos(Omega_k) - y_k_prime*np.cos(i_k)*np.sin(Omega_k)
  sv_position['y_k'] = x_k_prime*np.sin(Omega_k) + y_k_prime*np.cos(i_k)*np.cos(Omega_k)
  sv_position['z_k'] = y_k_prime*np.sin(i_k)
  return sv_position

# Rotate to correct ECEF satellite positions
def flight_time_correct(X, Y, Z, flight_time):
    theta = WE * flight_time/1e6
    R = np.array([[np.cos(theta), np.sin(theta), 0.], [-np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])

    XYZ = np.array([X, Y, Z])
    rot_XYZ = R @  np.expand_dims(XYZ, axis=-1)
    return rot_XYZ[0], rot_XYZ[1], rot_XYZ[2]