"""Calculate residuals

"""

__authors__ = "D. Knowles"
__date__ = "25 Jan 2022"

import numpy as np
import georinex as gr 
import nav_check_files as ncf
import utils_ephem as utils

SPEED_OF_LIGHT = 299792458
OmegaEDot = 7.2921151467e-5

def pseudoCorrections(gpstime, ephem):
    t_offset = gpstime - ephem['Toe'].values
    if np.abs(t_offset).any() > 302400:
        t_offset = t_offset-np.sign(t_offset)*604800

    # Calculate clock corrections from the polynomial
    # corr_polynomial = ephem.af0
    #                 + ephem.af1*t_offset
    #                 + ephem.af2*t_offset**2
    corr_polynomial = (ephem['SVclockBias'].values
                     + ephem['SVclockDrift'].values * t_offset
                     + ephem['SVclockDriftRate'].values * t_offset**2)
    
    clk_corr = (corr_polynomial - ephem['TGD'].values) # + corr_relativistic
    
    return clk_corr

def calc_dgps_corrections_snapshot(tidx, basestation_data, eph_gnss): 
    if basestation_data['raw_pr_m', tidx][0]<2e6:
        basestation_data["delta_pr_m", tidx] = np.nan
    else: 
        SATtransTime = np.array([basestation_data['raw_pr_m', tidx][0]/SPEED_OF_LIGHT])
        satXYZ_nav = ncf.extract_nav_satposvel(eph_gnss, basestation_data["gps_tow", tidx][0], \
                                               SATtransTime, \
                                               np.array([basestation_data["sv"][0, tidx]]))
#         print('before: ', satXYZ_nav)
        delX = (OmegaEDot * satXYZ_nav[:,1] * SATtransTime )
        delY = (-OmegaEDot * satXYZ_nav[:,0] * SATtransTime)
        satXYZ_nav_corrected = np.full_like(satXYZ_nav, 0)
        satXYZ_nav[:,0] = satXYZ_nav[:,0] + delX 
        satXYZ_nav[:,1] = satXYZ_nav[:,1] + delY 
#         print('after: ', satXYZ_nav)

        OBSbaseGTRUTH = np.hstack((basestation_data['x_gt_m', tidx], \
                                   basestation_data['y_gt_m', tidx], \
                                   basestation_data['z_gt_m', tidx]))

        exp_obs_pseudo = np.linalg.norm(OBSbaseGTRUTH-satXYZ_nav, axis=1)
        delta_pr_m = exp_obs_pseudo - basestation_data['raw_pr_m',tidx] 
#         eph_gnss_sv = eph_gnss.sel(sv=np.array([basestation_data["sv"][0, tidx]]))
#         SUBEPHidx = utils.findIdxs(basestation_data["gps_tow", tidx][0]-19, eph_gnss_sv)
#         eph_gnss_sv_time = eph_gnss_sv.sel(time=eph_gnss.indexes['time'][SUBEPHidx[0]])
#         delta_pr_m = exp_obs_pseudo - basestation_data['raw_pr_m', tidx] \
#                     - SPEED_OF_LIGHT * pseudoCorrections(basestation_data["gps_week", tidx], eph_gnss_sv_time)

        basestation_data["delta_pr_m", tidx] = delta_pr_m
        print(tidx, basestation_data["time"][0, tidx], \
                 basestation_data["sv"][0, tidx], \
                 basestation_data["gps_tow", tidx][0], \
                 basestation_data['raw_pr_m', tidx][0], \
                 basestation_data["delta_pr_m", tidx][0])

    return basestation_data


def calc_dgps_corrections(basestation_data, eph_input_path): 
#     eph_input_path = 'data_internal/ephemeris/nasa/brdc1180.21n' #'data_internal/slac118/slac1180.21n' #

    basestation_data["delta_pr_m"] = np.nan*np.ones(np.shape(basestation_data['raw_pr_m']))
    print(basestation_data["delta_pr_m"])

    eph_gnss = gr.load(eph_input_path) #brdc1360.20n: pixel4_derived_back 
    for i in range(len(basestation_data["raw_pr_m"][0])):
        if not (basestation_data["gnss_id",i][0]==1.0):
            continue
        if basestation_data['raw_pr_m', i][0]<2e6:
            continue
        SATtransTime = np.array([basestation_data['raw_pr_m',i][0]/SPEED_OF_LIGHT])
        satXYZ_nav = ncf.extract_nav_satposvel(eph_gnss, basestation_data["gps_tow",i][0]-19, \
                                               SATtransTime, \
                                               np.array([basestation_data["sv"][0,i]]))
        delX = (OmegaEDot * satXYZ_nav[:,1] * SATtransTime )
        delY = (-OmegaEDot * satXYZ_nav[:,0] * SATtransTime)
        satXYZ_nav_corrected = np.full_like(satXYZ_nav, 0)
        satXYZ_nav[:,0] = satXYZ_nav[:,0] + delX 
        satXYZ_nav[:,1] = satXYZ_nav[:,1] + delY 
#         satXYZ_nav_corrected[:,2] = satXYZ_nav[:,2] 
        
        OBSbaseGTRUTH = np.hstack((basestation_data['x_gt_m', i], \
                                   basestation_data['y_gt_m', i], \
                                   basestation_data['z_gt_m', i]))
        
        exp_obs_pseudo = np.linalg.norm(OBSbaseGTRUTH-satXYZ_nav, axis=1)
#         delta_pr_m = exp_obs_pseudo - basestation_data['raw_pr_m',i] 
        eph_gnss_sv = eph_gnss.sel(sv=np.array([basestation_data["sv"][0,i]]))
        SUBEPHidx = utils.findIdxs(basestation_data["gps_tow",i][0]-19, eph_gnss_sv)
        eph_gnss_sv_time = eph_gnss_sv.sel(time=eph_gnss.indexes['time'][SUBEPHidx[0]])
        delta_pr_m = exp_obs_pseudo - basestation_data['raw_pr_m',i] \
                     - SPEED_OF_LIGHT * pseudoCorrections(basestation_data["gps_week",i], eph_gnss_sv_time)

        basestation_data["delta_pr_m", i] = delta_pr_m
        print(i, basestation_data["time"][0,i], \
                 basestation_data["sv"][0,i], \
                 basestation_data["gps_tow",i][0], \
                 basestation_data['raw_pr_m', i][0], \
                 basestation_data["delta_pr_m", i][0])

#     print(basestation_data["delta_pr_m"])

    return basestation_data
