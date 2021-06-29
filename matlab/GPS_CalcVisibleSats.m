function [eph, elaz, sats]= GPS_CalcVisibleSats(ephem, gpsTime, guess, el_mask)

% Load constants
GPS_constants

% Estimate the ephemeris of all satellites

clear eph
sats = 1:32;
for j = 1:length(sats)
    %sats(j)
    % Find the ephem entries for this satellites
    ind = find(ephem(:,1) == sats(j));
    
    if (isempty(ind))
        continue; 
    end
    % Find the closest ephem entry if there are multiple ones
    [~,in] = min(abs(ephem(ind,5)-gpsTime)); %t_gps(end)
    
    % Save this ephem entry into a new array
    eph(j,:) = ephem(ind(in),:);
end

approx_XYZ = GPS_FindSat(eph, gpsTime-t_trans);
approx_elaz = GPS_elaz(guess, approx_XYZ(:,3:5)); % TODO: Look into replacing this with an LLA to ENU function
sats = sats(approx_elaz(:,1)>el_mask);
eph = eph(approx_elaz(:,1)>el_mask,:);
elaz = approx_elaz(approx_elaz(:,1)>el_mask,:);
end