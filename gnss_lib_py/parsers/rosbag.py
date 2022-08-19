import rosbag #http://wiki.ros.org/rosbag/Cookbook
import numpy as np

import matplotlib.pyplot as plt

bag_name = 'CA-20190828184706_blur_align.bag'
ublox_raw_topic_name = '/ublox_gps_node/rxmraw'
ground_truth_topic_name = '/novatel_data/bestpos'

#load bag
bag = rosbag.Bag(bag_name)

bag_dict = bag.get_type_and_topic_info()[1]
for k, v in bag_dict.items():
    print(f'Topic: {k}, Message count: {v[1]}')

#print sample ublox raw message
for topic, msg, t in bag.read_messages(topics=[ublox_raw_topic_name]):
    print(msg)
    break

#initialize arrays for: rcvTOW, week, leapS, prMes, cpMes, doMes, gnssId, svId, freqId, cno
raw_dict = {}
raw_dict['rcvTOW'], raw_dict['week'], raw_dict['leapS'] = np.empty(0, dtype='float'), np.empty(0, dtype='int'), np.empty(0, dtype='int')
raw_dict['prMes'], raw_dict['cpMes'], raw_dict['doMes'], raw_dict['cno'] = np.empty(0, dtype='float'), np.empty(0, dtype='float'), np.empty(0, dtype='float'), np.empty(0, dtype='float')
raw_dict['gnssId'], raw_dict['svId'], raw_dict['freqId'] = np.empty(0, dtype='object'), np.empty(0, dtype='int'), np.empty(0, dtype='int')
#freqId is only for Glonass, check pg. 411 of https://content.u-blox.com/sites/default/files/products/documents/u-blox8-M8_ReceiverDescrProtSpec_UBX-13003221.pdf

idx = 0 #measurement index
#iterate through all ublox raw msgs
for topic, msg, t in bag.read_messages(topics=[ublox_raw_topic_name]):
    #get number of measurements at current time
    numMeas = msg.numMeas

    #append empty arrays
    for k in raw_dict:
        raw_dict[k] = np.hstack((raw_dict[k], np.empty(numMeas, dtype=raw_dict[k].dtype)))

    #get rcvTOW, week, leapS to calculate satellite positions
    _rcvTOW, _week, _leapS = msg.rcvTOW, msg.week, msg.leapS

    #iterate over all measurements
    for iMeas in range(numMeas):
        #get gnssId, svId, freqId
        _gnssId, _svId, _freqId = msg.meas[iMeas].gnssId, msg.meas[iMeas].svId, msg.meas[iMeas].freqId
        #get required measurements
        _prMes, _cpMes, _doMes, _cno = msg.meas[iMeas].prMes, msg.meas[iMeas].cpMes, msg.meas[iMeas].doMes, msg.meas[iMeas].cno

        #update arrays
        raw_dict['rcvTOW'][idx], raw_dict['week'][idx], raw_dict['leapS'][idx] = _rcvTOW, _week, _leapS
        raw_dict['svId'][idx], raw_dict['prMes'][idx], raw_dict['cpMes'][idx], raw_dict['doMes'][idx], raw_dict['cno'][idx], raw_dict['freqId'][idx] = _svId, _prMes, _cpMes, _doMes, _cno, _freqId
        #gnssId assigned based on Section 10.9 in https://content.u-blox.com/sites/default/files/products/documents/u-blox8-M8_ReceiverDescrProtSpec_UBX-13003221.pdf (pg. 35 of 459)
        if _gnssId == 0: raw_dict['gnssId'][idx] = 'GPS'
        elif _gnssId == 1: raw_dict['gnssId'][idx] = 'SBAS'
        elif _gnssId == 2: raw_dict['gnssId'][idx] = 'Galileo'
        elif _gnssId == 3: raw_dict['gnssId'][idx] = 'Beidou'
        elif _gnssId == 4: raw_dict['gnssId'][idx] = 'IMES'
        elif _gnssId == 5: raw_dict['gnssId'][idx] = 'QZSS'
        elif _gnssId == 6: raw_dict['gnssId'][idx] = 'Glonass'
        else: raise ValueError('Unexpected gnssID: ' + raw_dict['gnssId'][idx]) #throw error for unexptected gnssId

        #update index
        idx += 1


#visualize measurement for single satellite
check_sv = 14
sv_idx = raw_dict['svId'] == check_sv
if np.any(sv_idx):
    sv_time = raw_dict['rcvTOW'][sv_idx] - raw_dict['rcvTOW'][0]
    sv_meas = raw_dict['prMes'][sv_idx]
    plt.plot( sv_time, sv_meas )
else:
    print(check_sv, 'does not exist in dataset', bag_name)


# ros_measure = Measurement()
# for k, v in raw_dict.items():
#     ros_measure[k] = v


#print sample NovAtel raw message
for topic, msg, t in bag.read_messages(topics=[ground_truth_topic_name]):
    print(msg)
    break


#initialize arrays for gpsWeekTime, gpsWeek, gt_lat, gt_lon, gt_alt, gt_lat_std, gt_lon_std, gt_alt_std based on n_gt_msgs
n_gt_msgs = bag_dict[ground_truth_topic_name][1]
gt_dict = {}
gt_dict['gpsWeekTime'], gt_dict['gpsWeek'] = np.empty(n_gt_msgs, dtype='float'), np.empty(n_gt_msgs, dtype='int')
gt_dict['gt_lat'], gt_dict['gt_lon'], gt_dict['gt_alt'] = np.empty(n_gt_msgs, dtype='float'), np.empty(n_gt_msgs, dtype='float'), np.empty(n_gt_msgs, dtype='float')
gt_dict['gt_lat_std'], gt_dict['gt_lon_std'], gt_dict['gt_alt_std'] = np.empty(n_gt_msgs, dtype='float'), np.empty(n_gt_msgs, dtype='float'), np.empty(n_gt_msgs, dtype='float')

idx = 0 #measurement index
#iterate through all ground truth msgs
for topic, msg, t in bag.read_messages(topics=[ground_truth_topic_name]):
    #get gps week Time
    gt_dict['gpsWeekTime'][idx], gt_dict['gpsWeek'][idx] = (msg.header.gps_week_seconds/1000), msg.header.gps_week

    #get (lat, lon, alt) and std
    gt_dict['gt_lat'][idx], gt_dict['gt_lon'][idx], gt_dict['gt_alt'][idx] = msg.latitude, msg.longitude, msg.altitude
    gt_dict['gt_lat_std'][idx], gt_dict['gt_lon_std'][idx], gt_dict['gt_alt_std'][idx] = msg.latitude_std, msg.longitude_std, msg.altitude_std

    #update index
    idx += 1


#visualize latitude variation
plt.plot(gt_dict['gt_lat'])

#close bag after reading
bag.close()
