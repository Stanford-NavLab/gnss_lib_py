a = 6378137;              % semi-major axis of the earth [m]
b = 6356752.3145;         % semi-minor axis of the earth [m]
e = sqrt(1-(b^2)/(a^2));  % eccentricity of the earth = 0.08181919035596
lat_accuracy_thresh = 1.57e-6; % 10 meter latitude accuracy
%leapSeconds = 18;         % GPS leap seconds [s]
muearth = 398600.5e9;     % G*Me, the "gravitational constant" for orbital
                          % motion about the Earth [m^3/s^2]
OmegaEDot = 7.2921151467e-5; % the sidereal rotation rate of the Earth
                          % (WGS-84) [rad/s]
c = 299792458;            % speed of light [m/s]
F = -4.442807633e-10;      % Relativistic correction term [s/m^(1/2)]
f1 = 1.57542e9;        % GPS L1 frequency [Hz]
f2 = 1.22760e9;         % GPS L2 frequency [Hz]
t_trans = 70*0.001; % ~70ms

