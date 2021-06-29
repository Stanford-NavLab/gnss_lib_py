function [Pr, doppler, satXYZ, satV] = GPS_SimulatedMeas(ephem, el_mask, gpsTime, gtruth, prange_std, doppler_std) 
    gtruth = reshape(gtruth, [1, length(gtruth)]);
    % breakdown the ground truth information
    obsLoc = gtruth(1:3);
    obsBias = gtruth(4);
    obsVel = gtruth(5:7);
    obsDrift = gtruth(8);

    %gT = [Gauss_T; Gauss_dT];

    % Load GPS constants
    GPS_constants;

    % Extract visible satellites

    [eph, ~, ~]= GPS_CalcVisibleSats(ephem, 219600, reshape(obsLoc, [1,3]), el_mask);

    [satXYZ, satV] = GPS_SatLocation(eph, gpsTime, obsLoc);

    % Create a 1 x 3 vector containing the ECEF vector from the current
    % guess to the satellites
    delXYZ = satXYZ-repmat(obsLoc, length(eph(:,1)),1);%[(satX-guessmatrix(:,1)) (satY-guessmatrix(:,2)) (satZ-guessmatrix(:,3))];

    range=sqrt(sum((delXYZ.^2)')'); % true range between receiver and satellites

    % Without pseudorange corrections
    % Pr = range + obsBias; 

    % With pseudorange corrections 
    corr_Pr = GPS_PseudoSatClock(eph, gpsTime, range);  
    Pr = range + obsBias-(corr_Pr(3:2:end)-range); 

    delV = satV-repmat(obsVel, length(eph(:,1)),1);
    Prdot = sum(delV.*delXYZ,2)./range;
    doppler = (f1/c)*(-Prdot-obsDrift); 

    if nargin > 4
        Pr = Pr + normrnd(0, prange_std, size(Pr));
        doppler = doppler + normrnd(0, doppler_std, size(doppler));
    end
end

% % Adding gaussian noise in pseudorange and doppler measurements
% Pr_mu = (Nbnds(1)-0)*rand(length(range),1)+0;
% Pr_sig = (Nbnds(2)-0)*rand(length(range),1)+0;
% 
% Dp_mu = (Nbnds(3)-0)*rand(length(range),1)+0;
% Dp_sig = (Nbnds(4)-0)*rand(length(range),1)+0;

% Add multipath effects to the pseudoranges
% elaz = GPS_elaz(obsLoc, satXYZ);
% [~, mpIdx] = sort(elaz(:,1));
% mpAvlb = zeros(size(range));
% Imp = mpIdx(1:n_mp);
% mpAvlb(Imp)=1;
% mpBias = mpAvlb.*((Mbnds(2)-Mbnds(1))*rand(length(range),1)+Mbnds(1));

%fPr = range+obsBias; % added the simulated ground truth clock bias
% GausN_Pr = sqrt(Pr_sig).*randn(size(range)) + Pr_mu;
% Fv = mpBias+GausN_Pr;
% GausN_Dp = sqrt(Dp_sig).*randn(size(range)) + Dp_mu;





