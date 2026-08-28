function y = remove_CP(x)
% Remove CP (Cyclic Prefix) of length Ng
global data YY
% Nps = 4;
% Np = (data*4)/Nps; % Number of pilots, 160
% N = (data*4) + Np; % 640

Ng = 640/4; %160
Noff=0;
%x=reshape(x,1,800); %edit sini kalau jawapan salah
%x=x';
    y = x(:,Ng+1-Noff:Ng+640); %output y=[1x640]
%YY=y;%1x640