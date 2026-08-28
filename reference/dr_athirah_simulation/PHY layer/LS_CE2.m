function H_LS = LS_CE2(u)

global  xp  pilot_loc int_opt LS_est Nfft method H_LS_all uu
Nfft=640;
Nps=4;
Np=length(pilot_loc)/4;
uu = u;
method='linear';
xp_temp = reshape(xp',160,4)';
%Y = reshape(u',160,Nps)';
%Yp =[]; h = [];Re_Yp=[];Im_Yp=[];
for user=1:Nps
for k=1:Np
LS_est(user,k) = u(user,pilot_loc(k))./xp_temp(pilot_loc(k)); % LS channel estimation
end
end
% if lower(int_opt(1))=='l'
%     method='linear';
% else
%     method='spline';
% end
LS_est_temp= reshape(LS_est',128,1)';
for q =1:160
% Linear/Spline interpolation

%H_LS(user,:) = interpolate(LS_est(user,:),pilot_loc(1:Np),160,method); 
HData_LS(:,q) = interpolate(LS_est_temp(:,q).',pilot_loc,160,'linear');
end
%H_LS_all=[H_LS(1,:),H_LS(2,:),H_LS(3,:),H_LS(4,:)]';
out = H_LS;

