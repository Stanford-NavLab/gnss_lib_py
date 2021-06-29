function pseudo_range = GPS_PseudoSatClock(ephem, gpsTime, pseudo)
%
% < DOES NOT INCLUDE IONOSPHERIC CORRECTIONS >
%
% Function to add the satellite clock corrections back into the
% pseudoranges recorded by the GPSRCVR program in the RINEX format.
%
% Inputs:
%       ephem - an M x 27 array containing the complete ephemeris
%               information for each satellite (if they are available).  See
%               GPS_ParseEphem.m for a description of the array format.  The size
%               of ephem should be matched to the time vector (see description for
%               'gpsTime' below).
%       gpsTime - the GPS time to compute the satellite position(s) for.
%       pseudo - an M x N array containing N time samples of the
%                pseudoranges to the M satellites in the ephem matrix.
%
% Outputs:
%       pseudo_range - a (N x 2M + 1) array containing the GPStime, and then pairs
%               of svID and un-corrected pseudo_ranges [m]:
%               range = [GPStime svID pr svID pr ... ;
%                          . . . . .
%                        GPStime svID pr svID pr ...];
%
% Revision history:
%       18-Oct-2003: Initial creation at Cornell University
%       03-Aug-2006: Algorithm converted for use at University of Illinois
%       by Jonathan J. Makela (jmakela@uiuc.edu)
%		06-Aug-2006: Added SVid to ephem array by Jonathan J. Makela
%		(jmakela@uiuc.edu)

% Load GPS constants
GPS_constants;

% If multiple times input, make sure the vector is oriented correctly
if(length(gpsTime) > 1)
    if(length(gpsTime) ~= size(gpsTime,1))
        gpsTime = gpsTime';
    end
end

% Determine the number of satellite ranges being calcualted
satellites = size(ephem,1);

% Get clock correction parameters from ephem
TOC = ephem(:,7);   % Reference time of clock, TOC [s]
af0 = ephem(:,10);      % Satellite clock correction polynomial, af0 [s]
af1 = ephem(:,11);      % Satellite clock correction polynomial, af1 [s/s]
af2 = ephem(:,12);      % Satellite clock correction polynomial, af2
% [s/s/s]
tgd = ephem(:,9);       % Estimated group delay differentail, Tgd [s]

% Terms needed for the relativisitc correction
ecc = ephem(:,13);      % Eccentricity
sqrta = ephem(:,17);    % Square root of semimajor axis [m^1/2]
M0 = ephem(:,21);       % Mean anomaly, M0 [rad]
dn = ephem(:,18);       % Mean motion difference [1/s]

% Initialize the output matrix with the time variable
pseudo_range = gpsTime';

% Create pseudo_range by correcting raw pseudorange measurements for each
% time sample.  Do this on a per satellite basis, so make sure to reference
% the correct satellite when needed (e.g., all of the terms from the
% ephemerides.
for sv = 1:satellites
    % Determine pseudorange corrections due to satellite clock corrections.
    % Calculate time offset from satellite reference time
    timeOffset = gpsTime - TOC(sv);
    if(abs(timeOffset) > 302400)
        timeOffset = timeOffset-sign(timeOffset).*604800;
    end

    % Calculate mean anomaly with corrections
    Mcorr = dn(sv) .* timeOffset;
    M = M0(sv) + (sqrt(muearth) * (sqrta(sv)).^(-3)) .* timeOffset + Mcorr;

    % Compute the eccentric anomaly from mean anomaly using the Newton Raphson
    % method to solve for 'E' in:
    %   f(E) = M - E + ecc * sin(E) = 0
    E = M;
    for i = 1:10
        f = M - E + ecc(sv) .* sin(E);
        dfdE = - 1 + ecc(sv) .* cos(E);
        dE = - f ./ dfdE;
        E = E + dE;
    end

    % Calculate clock corrections from the polynomial
    corrPolynomial = af0(sv) + af1(sv) .* timeOffset + af2(sv) .* timeOffset.^2;

    % Calcualte the relativistic clock correction
    corrRelativistic = F .* ecc(sv) .* sqrta(sv) .* sin(E);

    % Calculate the total clock correction including the Tgd term
    clockCorr = (corrPolynomial - tgd(sv) - corrRelativistic);

    % Calculate total pseudorange correction
    pseudoCorr = clockCorr .* c;

    % apply corrections to pseudorange measurements and create output array
    pseudo_range = [pseudo_range;[ephem(sv,1)*ones(size(pseudoCorr)) pseudo(sv,:)'+pseudoCorr]'];
end