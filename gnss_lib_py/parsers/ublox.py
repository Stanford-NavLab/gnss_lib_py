import numpy as np
import pandas as pd
from pyubx2 import UBXReader



def parse_navposecef(data_string):

    # get start indices of message fields
    comma_inds = [pos for pos, char in enumerate(data_string) if char==',']
    itow_ind = data_string.find('iTOW')
    ecefx_ind = data_string.find('ecefX')
    ecefy_ind = data_string.find('ecefY')
    ecefz_ind = data_string.find('ecefZ')

    # extract 
    iTOW = data_string[itow_ind + 5 : comma_inds[1]]
    ecefx = data_string[ecefx_ind + 6 : comma_inds[2]]
    ecefy = data_string[ecefy_ind + 6 : comma_inds[3]]
    ecefz = data_string[ecefz_ind + 6 : comma_inds[4]]

    return iTOW, ecefx, ecefy, ecefz 


stream = open('static_cones/CenterConeSurveyStatic.ubx', 'rb')
ubr = UBXReader(stream, protfilter=2)
cone_loc_ecef = []

read = True
while read:
    try:
        raw_data, parsed_data = ubr.read()
        if parsed_data.identity == 'NAV-POSECEF':
            iTOW, ecefx, ecefy, ecefz = parse_navposecef(str(parsed_data))
            cone_loc_ecef.append([iTOW, ecefx, ecefy, ecefz])
        
    except:
        break

cone_loc_ecef = np.array(cone_loc_ecef)
df = pd.DataFrame(cone_loc_ecef)
df.to_csv('center_cone_ecef.csv')

