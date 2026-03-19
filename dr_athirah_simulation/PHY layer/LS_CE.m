function [H_LS] = LS_CE(Y,Xp,pilot_loc,Nfft,Nps,int_opt)
global LS_est
% LS channel estimation function
% Inputs:
% Y = Frequency-domain received signal
% Xp = Pilot signal
% pilot_loc = Pilot location
% N = FFT size
% Nps = Pilot spacing
% int_opt = ’linear’ or ’spline’
% output:
% H_LS = LS Channel estimate
Np=512/4; k=1:128;
LS_est(k) = Y(pilot_loc(k))./Xp(k);% LS channel estimation
if lower(int_opt(1))=='l', method='linear'; 
else method='spline'; 
end
% Linear/Spline interpolation
 H_LS = interpolate(LS_est,pilot_loc,Nfft,method);
%  H_LS=zeros(2048,2048);
%  for i=1:2048
%      H_LS(i,i)=H_i(i);
%  end
% % H_LS=LS_est;