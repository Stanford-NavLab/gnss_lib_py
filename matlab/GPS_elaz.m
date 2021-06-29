function [elaz] = GPS_elaz(Rx, Sats)
%
% Function to calculate the elevation and azimuth from a single receiver to
% satellites.
%
% Inputs:
%       Rx - a (1x3) vector containing the ECEF X, Y, and Z coordinates of
%       the receiver in meters.
%
%       Sats - a (N x 3) array containing the X, Y, and Z locations in ECEF
%       coordinates for the N satellites being considered.  X, Y, and Z are
%       given in meters.  There are N different coordinates.
%               xyz = [X_Sat0, Y_Sat0, Z_Sat0;
%                      X_Sat1, Y_Sat1, Z_Sat1;
%                       . . . . . . . . . . .
%                      X_SatN, Y_SatN, Z_SatN];
%
% Outputs:
%       elaz - a (N x 2) array containing the elevation and azimuth from the
%       receiver to the requested satellites.  Elevation and azimuth are
%       given in decimal degrees.  There are N different coordinates to be 
%       transformed.
%               elaz = [el0, az0;
%                      el1, az1;
%                       . . . . 
%                      azN, azN]
%
% Revision history:
%       11-Jun-2006: Initial template created by Jonathan J. Makela
%       (jmakela@uiuc.edu)
%       19-Jun-2006: Code developed by Dwayne Hagerman
%       20-Jun-2006: Revised code to make more matrix friendly by Jonathan
%       J. Makela (jmakela@uiuc.edu)

% Convert the receiver location to WGS84
lla = ecef2lla(Rx); %GPS_WGS84(Rx);

% Create variables with the latitude and longitude in radians
lat = lla(1) * (pi()/180);
long = lla(2) * (pi()/180);

% Create the 3 x 3 transform matrix from ECEF to VEN
VEN = [ cos(lat)*cos(long),     cos(lat)*sin(long),     sin(lat);
                -sin(long),              cos(long),            0;
       -sin(lat)*cos(long),    -sin(lat)*sin(long),     cos(lat);];

% Replicate the Rx array to be the same size as the satellite array
Rx_array = ones(size(Sats,1),1) * Rx;

% Calculate the pseudorange for each satellite
p = Sats - Rx_array;

% Calculate the length of this vector
n = sqrt(p(:,1).^2 + p(:,2).^2 + p(:,3).^2);

% Create the normalized unit vector
p = p ./ [n n n];

% Perform the transform of the normalized psueodrange from ECEF to VEN
p_VEN = VEN * p';

% Calculate elevation and azimuth in degrees
elaz(:,1) = (pi()/2. - acos(p_VEN(1,:))) * 180./pi();
elaz(:,2) = atan2(p_VEN(2,:),p_VEN(3,:)) * 180./pi();