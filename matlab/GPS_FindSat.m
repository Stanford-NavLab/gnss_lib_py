function satLoc = GPS_FindSat(ephem, t)
%
% Function to calculate satellite positions given the orbital ephemerides.
% The calculation can either be done for one satellite and multiple times,
% or for multiple satellites at one time.
%
% Inputs:
%       ephem - an M x 27 array containing the complete ephemeris
%       information for each satellite (if they are available).  See
%       GPS_ParseEphem.m for a description of the array format.
%       t - the GPS time to compute the satellite position(s) for. t can
%           can take on two forms:
%               1) if t is a N x 1 vector, than the satellite position will be
%               calculated at all times specified by t.  In this case,
%               ephem will be a 1 x 27 vector containing the ephemerides
%               for ONLY ONE satellite.
%               2) if t is a scalar, than the satellite positions of
%               multiple satellites at the single time specified by t will
%               be calculated.  In this case, ephem will be a M x 27 array
%               containing the ephermerides to ALL satellites for which the
%               position should be calculated.
%
% Outputs:
%       satLoc - a (N x 5) array containing the GPStime, X, Y, and Z locations in ECEF
%       coordinates calculated from the ephemerides.  X, Y, and Z are
%       given in meters.  GPStime is in seconds since the midnight transition from the
%       previous Saturday/Sunday.  There are N different solutions,
%       determined either by the length of the 't' vector (if one satellite and multiple
%       time steps) or the number of rows in the 'ephem' matrix (if multiple satellites and a
%       single time)
%               satLoc = [GPStime svID ECEFx ECEFy ECEFz ;
%                          . . . . .
%                         GPStime svID ECEFx ECEFy ECEFz];
%
% Revision history:
%       31-Aug-2001: Initial creation at Cornell University
%       02-Aug-2006: Algorithm converted for use at University of Illinois
%       by Jonathan J. Makela (jmakela@uiuc.edu)
%		06-Aug-2006: Added svID to satLoc and ephem arrays by Jonathan J.
%		Makela (jamkela@uiuc.edu)

% Load GPS constants
GPS_constants;

% Determine the number of satellites contained in the ephem variable
satellites = size(ephem,1);
if(satellites == 0)
    return;
end

% Parse the ephem matrix to get the different orbital parameters
t0 = ephem(:,5);        % Reference time of ephemeris [s]
ecc = ephem(:,13);      % Eccentricity
sqrta = ephem(:,17);    % Square root of semimajor axis [m^1/2]
omega = ephem(:,20);    % Argument of perigee [rad]
M0 = ephem(:,21);       % Mean anomaly, M0 [rad]
Omega0 = ephem(:,19);   % Longitude of ascending node at TOE [rad]
OmegaDot = ephem(:,16); % Rate of right ascension [rad/s]    
dn = ephem(:,18);       % Mean motion difference [1/s]
i0 = ephem(:,14);       % Orbital inclination [rad]
iDot = ephem(:,15);     % Rate of inclination angle [rad/s]
cuc = ephem(:,22);      % [rad]
cus = ephem(:,23);      % [rad]
crc = ephem(:,24);      % [m]
crs = ephem(:,25);      % [m]
cic = ephem(:,26);      % [rad]
cis = ephem(:,27);      % [rad]

% Check the size of 't'.  If it is a vector, than we are calculating the
% satellite position of a single satellite over time.  Only one satellite
% is allowed in this case.  Return if more than one satellite is requested.
% If it is a scalar than we are calculating the position of multiple
% satellites at a single time
% if((length(t) ~= 1) && (satellites ~= 1))
%     return;
% end
% 
% If multiple times input, make sure the vector is oriented correctly
 if(length(t) > 1)
     if(length(t) ~= size(t,1))
         t = t;
     end
 end

% If 't' is a single value, create the time vector
if(size(t,1) == 1)
    t = t.*ones(satellites,1);
end

% Define time of position request and delta t from epoch.  Correct for
% possible week crossovers.  Note that there are 604800 seconds in a GPS
% week
dt = t - t0;
% % idx = find(dt > 302400);    % Finds times in the next week
% % dt(idx) = dt(idx) - 604800;
% % idx = find(dt < -302400);   % Finds times in the previous week
% % dt(idx) = dt(idx) + 604800;

% Calculate mean anomaly with corrections
Mcorr = dn .* dt;
M = M0 + (sqrt(muearth) * (sqrta).^(-3)) .* dt + Mcorr;

% Compute the eccentric anomaly from mean anomaly using the Newton Raphson
% method to solve for 'E' in:
%   f(E) = M - E + ecc * sin(E) = 0
E = M;
for i = 1:10
    f = M - E + ecc .* sin(E);
    dfdE = - 1 + ecc .* cos(E);
    dE = - f ./ dfdE;
    E = E + dE;
end

% Calculate true anomaly from eccentric anomaly
sinnu = sqrt(1 - ecc.^2) .* sin(E) ./ (1 - ecc .* cos(E));
cosnu = (cos(E) - ecc) ./ (1 - ecc .* cos(E));
nu = atan2(sinnu,cosnu);

% Calculate the argument of latitude iteratively.
phi0 = omega + nu;
phi = phi0;
for i = 1:5
    cos2phi = cos(2*phi);
    sin2phi = sin(2*phi);
    phiCorr = cuc .* cos2phi + cus .* sin2phi;
    phi = phi0 + phiCorr;
end

% Calculate longitude of ascending node with correction
OmegaCorr = OmegaDot .* dt;
Omega = Omega0 - OmegaEDot .* t + OmegaCorr;

% Calculate orbital radius with correction
rCorr = crc .* cos2phi + crs .* sin2phi;
r = (sqrta.^2) .* (1 - ecc .* cos(E)) + rCorr;

% 1: New lines added for satellite velocity
dE = (sqrt(muearth) * (sqrta).^(-3)+dn)./(1-ecc .* cos(E));
dphi=sqrt(1-(ecc.^2)).*dE./(1-ecc.*cos(E));
dr= (sqrta.^2) .* ecc .*sin(E) .*dE + 2 * (crs.*cos2phi - crc.*sin2phi).* dphi;

% Calculate inclination with correction
iCorr = cic .* cos2phi + cis .* sin2phi + iDot .* dt;
i = i0 + iCorr;

% 2: New lines added for satellite velocity
di = 2 * (cis .* cos2phi - cic .* sin2phi).* dphi + iDot;

% Find position in orbital plane
xp = r .* cos(phi);
yp = r .* sin(phi);

% 3: New lines added for satellite velocity
du=(1+2*(cus .*cos2phi-cuc .*sin2phi)).* dphi;
dxp = dr .* cos(phi) - r .* sin(phi) .* du;
dyp = dr .* sin(phi) + r .* cos(phi) .* du;

% Find satellite position in ECEF coordinates
ECEFx = (xp .* cos(Omega)) - (yp .* cos(i) .* sin(Omega));
ECEFy = (xp .* sin(Omega)) + (yp .* cos(i) .* cos(Omega));
ECEFz = (yp .* sin(i));
%old line: satLoc = [t ephem(:,1).*ones(size(t,1),1) ECEFx ECEFy ECEFz];

dOmega = OmegaDot - OmegaEDot;
Vx = dxp .* cos(Omega) -  dyp .* cos(i) .* sin(Omega) ...
     + yp .* sin(Omega) .* sin(i) .* di ...
     - (xp .* sin(Omega) + yp .* cos(i) .* cos(Omega)) .* dOmega;
Vy = dxp .* sin(Omega) + dyp .* cos(i) .* cos(Omega) ...
     - yp .* sin(i) .* cos(Omega) .* di ...
     + (xp .* cos(Omega) - yp .* cos(i) .* sin(Omega)) .* dOmega;
Vz = dyp .* sin(i) + yp .* cos(i) .* di;

satLoc = [t ephem(:,1).*ones(size(t,1),1) ECEFx ECEFy ECEFz Vx Vy Vz];

