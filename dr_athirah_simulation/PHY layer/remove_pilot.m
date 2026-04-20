function out = remove_pilot(u)
global data_loc;
% Get frequency-domain signal data, Y
%uu=reshape(u',640,1)';
uu =reshape(u,1,640);
%Y = reshape(u,1,640);
Data_temp = uu(data_loc);
Data=reshape(Data_temp',128,4)';
%Data=reshape(Data_temp,4,128);
out = Data;
%Data = reshape(Data_temp,1,512);%1x512
% Rec_d_LS=Y(:,data_loc);
% Data = Rec_d_LS;
