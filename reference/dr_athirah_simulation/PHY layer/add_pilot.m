function out=add_pilot(u)
% CAZAC (Constant Amplitude Zero AutoCorrelation) sequence –> pilot
% Nps : Pilot spacing
global pilot_loc data_loc Xp xp xp_out

x = reshape(u',512,1)';
%x = reshape(u,1,512);
Nps = 4;
Np = length(x)/Nps; % Number of pilots = 128
N = length(x) + Np; % 512+128 = 640
%u_pilot = reshape(u',Np,Nps)';
% Add pilot
%xp = zeros(Nps,Np);
xp=[];

% for k = 1:(Np+Np/4)
%     xp((k-1)*Nps + 1) = 1;%exp(1j*pi*(k-1)^2/N); % Eq.(7.17) for Pilot boosting
% end


% xp = sqrt(real(xp).^2 + imag(xp).^2);


% Add frequency-domain signal data

pilot_loc=[];
data_loc=[];
Xp=[]; %pilot value
%for row=1:Nps
    j = 0;
for m = 1:N
    if mod(m,Nps+1) == 1
        % Add Pilot
        xp(m) = 1;
        pilot_loc = [pilot_loc m]; %PILOT LOCATION
        Xp =[Xp xp(m)]; 
        j = j+1;
    else
        %Add data after pilot
        xp(m) = x(m-j); % DATA + PILOT
        data_loc = [data_loc m]; %DATA LOCATION
    end
end

%end

%xp_1D =[xp(1,:),xp(2,:),xp(3,:),xp(4,:)];

 %xp_out=reshape(xp',128,5)';
 %w = cat(2,xp(1,:),xp(2,:),xp(3,:),xp(4,:),xp(5,:));
%out=w; %pilot + data 1x640
%xp_1D = reshape (xp',160,4)';
out=xp;
% pilot_loc = find(xp==1);
% data_loc = find(~(xp==1));

