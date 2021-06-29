function [satXYZ, satV] = GPS_SatLocation(ephem, gpsTime, obsLoc)

% Load GPS constants
GPS_constants;

satellites = length(ephem(:,1));
satXYZ = GPS_FindSat(ephem,gpsTime-t_trans);
delXYZ = satXYZ(:,3:5)-repmat(obsLoc, satellites,1);
range=sqrt(sum((delXYZ.^2)')');

% Calculate the transmission time to each satellite from the current
% guess location
tcorr = range'./c;

% Find the satellites at this time
% satXYZ = GPS_FindSat(ephem, gpsTime-tcorr); 
    
satV = satXYZ(:,6:8);
satX = satXYZ(:,3);
satY = satXYZ(:,4);
satZ = satXYZ(:,5);

% Calculate the position correction for the Earth's rotation
delX = OmegaEDot*satY'.*tcorr; 
delY = -OmegaEDot*satX'.*tcorr;
satXYZ = satXYZ(:,3:5) + [ delX' delY' zeros(satellites,1)];
end