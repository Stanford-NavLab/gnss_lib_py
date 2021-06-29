function [eph, sats, cPr, cDp] = GPS_RecvNoise(eph, sats, Pr, Dp, Measbnds)

    Pr_mu = (Measbnds(1)-0); %*rand(length(Pr),1)+0;
    Pr_sig = (Measbnds(2)-0); %*rand(length(Pr),1)+0;
    
    GausN_Pr = sqrt(Pr_sig).*randn(size(Pr)) + Pr_mu.*rand(size(Pr));
    cPr = Pr + GausN_Pr;
    
    Dp_mu = (Measbnds(3)-0); %*rand(length(Dp),1)+0;
    Dp_sig = (Measbnds(4)-0); %*rand(length(Dp),1)+0;

    GausN_Dp = sqrt(Dp_sig).*randn(size(Dp)) + Dp_mu.*rand(size(Dp));
    cDp = Dp + GausN_Dp; 
      
end