% Nor Fadzilah Abdullah
% University of Bristol, UK
% Feb 2017
% Fountain codes for safety broadcast

%% initialisation
%clear global; clear all; clc;
clear all;
hold on
rand('state', sum(100*clock));
randn('state', sum(100*clock));
scen = 'moving'; % 'stationary' or 'moving'
phyType = 'IEEE802.11';
MCS = 4;
tao_rms = 7.98972905957126e-09; %change according to different envi
Fd = 610.231145966653;
%phyType='DSRC'; MCS=3; % assumed fixed MCS mode for safety
Ptx = 29; %29;% transmit power (dBm) %29 for 3.5GHz (change accordingly)
Gt = 0; %dB
Gr = 8; %dB
fc = 3.5e9; %3.5e9; % 28GHz and 3.5GHz
ant = 'siso';
ksi = 0; % Full duplex =1; Half Duplex=0;
res = 1; % res=1 all V2V resources available; res=0.5 half of the resources available
%% RaptorQ parameters

SB = 512; % source block (bytes)
SS = 64; % source symbols (bytes)
K = SB / SS;

%% Mobility model (Cellular Automata: microscopic flow model)

disp('Block 1: Mobility model (CFM)') %indoor factory design, change accordingly
tic

tStep = 1e-3; %100e-3; % 100ms 
simTime = 1;%10 average runs

switch scen
    case 'stationary'
        FactoryLength = 1000;
        FactoryLaneWidth =15;
        ConveyorWidth = 3; % meter
        numSenPerLaneKM =25; % sparse (25 sen/km/lane), moderate (50 sen/km/lane), dense (100 sen/km/lane);
        numLane = 3;
        laneY = [0, 15, 18, 33, 36, 51, 54, 69].'; % 6 lanes/direction
        numSenPerLane = 25; % sparse (25 sen/km/lane), moderate (50 sen/km/lane), dense (100sen/km/lane);
        SenDensityKM = numSenPerLaneKM/1000*numLane;%par.density=[12,36,66]/1000;
        SenDensity = SenDensityKM*FactoryLength; %(sparse =4, moderate=5, dense=5/6)
        %laneX =[11, 30, 50, 50, 70];
        %laneX =[0:16:80];
        %laneX =[3, 6.7, 13.33, 20, 23, 26.7, 33.4, 40.1, 46.8, 53.53, 57, 60.2, 67, 73.63, 80];
        %laneX = [0:4:80];
        
        
        meanInterD = 1000/numSenPerLaneKM;
        temp = poissrnd(meanInterD,[numLane,numSenPerLaneKM]);
        Sensor_xLocation = mod(cumsum(temp,2),FactoryLength);
        Sensor_xLocation = sort(Sensor_xLocation,2);
        Sensor_yLocation = repmat(laneY,1,numSenPerLane);
        Sensor = struct('xLocation',0,'yLocation',0,'velocity',0);
        
        v_min = 0;
        v_max = [1 1 1]; %[60 90 120]; %km/h
        Length = FactoryLength;
        startVel = [randi([v_min v_max(1)],numLane,numSenPerLane);...
            randi([v_max(1) v_max(2)],numLane,numSenPerLane)];
        startVel = [startVel(2:end,:); startVel(1,:)];
        t = 1;
        k = 1;
        for j = 1:numLane
            for i = 1:numSenPerLane
                sensor(k).velocity(t) = startVel(j,i);
                sensor(k).xLocation(t) = Sensor_xLocation(j,i);
                sensor(k).yLocation(t) = Sensor_yLocation(j,i);
                k = k + 1;
            end
        end
    case 'moving'
        FactoryLength = 1000;
        FactoryLaneWidth =15;
        ConveyorWidth = 3; % meter
        numSenPerLaneKM = 100; % sparse (25 sen/km/lane), moderate (50 sen/km/lane), dense (100 sen/km/lane);
        numLane = 3;
        laneY = [0, 15, 18, 33, 36, 51, 54, 69].'; % 6 lanes/direction
        numSenPerLane = 100; % sparse (25 sen/km/lane), moderate (50 sen/km/lane), dense (100sen/km/lane);
        SenDensityKM = numSenPerLaneKM/1000*numLane;%par.density=[12,36,66]/1000;
        SenDensity = SenDensityKM*FactoryLength; %(sparse =4, moderate=5, dense=5/6)
        %laneX =[11, 30, 50, 50, 70];
        %laneX =[0:16:80];
        %laneX =[3, 6.7, 13.33, 20, 23, 26.7, 33.4, 40.1, 46.8, 53.53, 57, 60.2, 67, 73.63, 80];
        %laneX = [0:4:80];
        
        
        meanInterD = 1000/numSenPerLaneKM;
        temp = poissrnd(meanInterD,[numLane,numSenPerLaneKM]);
        Sensor_xLocation = mod(cumsum(temp,2),FactoryLength);
        Sensor_xLocation = sort(Sensor_xLocation,2);
        Sensor_yLocation = repmat(laneY,1,numSenPerLane);
        Sensor = struct('xLocation',0,'yLocation',0,'velocity',0);
        
        v_min = 10;
        v_max = [30 70 100]; %[60 90 120]; %km/h
        Length = FactoryLength;
        startVel = [randi([v_min v_max(1)],numLane,numSenPerLane);...
            randi([v_max(1) v_max(2)],numLane,numSenPerLane)];
        startVel = [startVel(2:end,:); startVel(1,:)];
        t = 1;
        k = 1;
        for j = 1:numLane
            for i = 1:numSenPerLane
                sensor(k).velocity(t) = startVel(j,i);
                sensor(k).xLocation(t) = Sensor_xLocation(j,i);
                sensor(k).yLocation(t) = Sensor_yLocation(j,i);
                k = k + 1;
            end
        end
end


switch scen
    case 'stationary'
        for t = 2:simTime
            t
            for k = 1:SenDensity
                if k <= numSenPerLane || k > (numLane/2)*numSenPerLane
                    vmax_j = v_max(1);
                else
                    vmax_j = v_max(2);
                end
                switch mod(k,numSenPerLane)
                    case 0
                        j = k/numSenPerLane;
                        precedingSenLoc = sensor((j-1)*numSenPerLane+1).xLocation(t-1);
                    otherwise
                        precedingSenLoc = sensor(k+1).xLocation(t-1);
                end
                delta_x = abs(sensor(k).xLocation(t-1)-precedingSenLoc)/tStep*3.6; % in km/h
                sensor(k).velocity(t) = min([vmax_j,sensor(k).velocity(t-1),delta_x]); % accelerate/decelerate
                sensor(k).xLocation(t) = mod(sensor(k).xLocation(t-1)+sensor(k).velocity(t),Length);
                sensor(k).yLocation(t) = mod(sensor(k).yLocation(t-1)+sensor(k).velocity(t),Length);
            end
        end
    case 'moving'
        for t=2:simTime
            t
            for k=1:SenDensity
                if k<=numVehPerLane || k>(numLane/2)*numVehPerLane
                    vmax_j=v_max(1);
                else
                    vmax_j=v_max(2);
                end
                switch mod(k,numVehPerLane)
                    case 0, j=k/numVehPerLane;
                        precedingVehLoc=vehicle((j-1)*numVehPerLane+1).xLocation(t-1);
                    otherwise, precedingVehLoc=vehicle(k+1).xLocation(t-1);
                end
                delta_x=abs(vehicle(k).xLocation(t-1)-precedingVehLoc)/tStep*3.6; % in km/h
                vehicle(k).velocity(t)=min([vmax_j,vehicle(k).velocity(t-1),delta_x]); % accelerate/decelerate
                vehicle(k).xLocation(t)=mod(vehicle(k).xLocation(t-1)+vehicle(k).velocity(t),Length);
                vehicle(k).yLocation(t)=mod(vehicle(k).yLocation(t-1)+vehicle(k).velocity(t),Length);
            end
        end
end

%% PHY layer Implementation
disp('Block 2: PHY implementation (SNR vs. PER)')
tic
% LTE-V MCS peak data rates
No_sub_per_user = 600;
Table = [5.6   1/3 2 -82; % QPSK
    6.3   3/8 2 -82;
    7.2   3/7 2 -82;
    8.4   1/2 2 -82; % *** choose this
    10.08 3/5 2 -80;
    12.6  3/4 2 -80;
    14.4  6/7 2 -80;
    11.2  1/3 4 -77; % 16QAM
    12.61 3/8 4 -77;
    14.41 3/7 4 -77;
    16.8  1/2 4 -77;
    20.16 3/5 4 -73;
    25.2  3/4 4 -73;
    21.6  3/7 6 -73; % 64QAM
    25.21 1/2 6 -69;
    30.24 3/5 6 -69;
    33.6  2/3 6 -69;
    37.8  3/4 6 -68;
    43.2  6/7 6 -68];

Rd = Table(MCS,1);
Rc = Table(MCS,2);
N_c = Table(MCS,3);               % Coded_bit_symbol

N_d = (No_sub_per_user*N_c*Rc); 	% Coded_data
N_s = (N_d/Rc);                   % Coded_bits
B = 0.22e6; %1e6; %0.22e6 for 3.5GHz;
NF = 7;%7;%2; % dB (UL: 2, DL=7)% changed this from 7 to 9 and distance reduces
% noise power
% boltzmannK=1.3806485279e-23;
thermalNoise = -174;%-174; % = 10*log10(boltzmannK*290*10^3*1000)-30 dBM/Hz;
noiseP = thermalNoise + 10 * log10(B) + NF; % in dBm
lambda = 3e8/fc;
switch scen
    case 'stationary'
        snr = 2.7;
        d_sense = 0;
        PLexp =3.6; %3.6 for 3.5Ghz, 3.1 for 28GHz %3.25, 3.48 for mmse perfect
        d_th = 0;
    case 'moving'
        snr=2.7;
        d_sense=100;
        PLexp=3.6;%3.43; for all except mmse perfect
        d_th=360;
        %         snr = 3;
        %         d_sense = 100;
        %         PLexp = 2.7;
        %         d_th = 100;
end
%RxSen = thermalNoise + 10 * log10(B) + NF + 5 + snr - Gr;%Table(MCS,4);5dB impementation loss, snr=2.2 dB for 0.01 per
%RxSen = thermalNoise + 10 * log10(B) + NF + 5 + snr - Gr;
RxSen = -105.4; %-105.4;
%REFSENSV2V=kTB + SNRV2V +10log10(LCRB/NRB) +( NFV2V+ IM) %refer 36.885 Rel 14 LTE test spec

% discrete-time simulation
for t = 1:simTime % translate dist->SNR and find out FEC performance
    for k = 1:SenDensity
        %         vehicle(k).intf=zeros(vehDensity,simTime);
        % find SNR
        for kk = 1:SenDensity
            sensor(k).dist(kk,t) = sqrt((sensor(k).xLocation(t)-sensor(kk).xLocation(t))^2 + (sensor(k).yLocation(t)-sensor(kk).yLocation(t))^2);
            if sensor(k).dist(kk,t) == 0
                sensor(k).dist(kk,t) = 0.001;
            end
            sensor(k).Prx(kk,t) = Ptx + Gt + Gr + 20 * log10(lambda/4/pi) + 10 *log10((1/sensor(k).dist(kk,t))^PLexp);
        end
        temp = find(~isinf(sensor(k).Prx(:,t))&sensor(k).Prx(:,t)>RxSen);
        %idx=(rand(length(temp),1) < 0.1 ); temp = temp(idx); %(10% simulataneous V2V tx)
        sensor(k).intf(1:numel(temp),t) = temp;
        sensor(k).intfnum(1,t) = numel(temp);
        for kk = 1:SenDensity
            sensor(k).SNR(kk,t) = sensor(k).Prx(kk,t)-10*log10(10^((noiseP+ksi*Ptx)/10)+ 10^(sum(sensor(k).Prx(temp,t))/10));
        end
        
        pktsize = [50 100 300 500];
        for ii = 1:length(pktsize)
            switch scen
                case 'stationary'
                    %filename = ['D:\Documents\From Pendrive Digi\Simulator\V2V fizi\Fizi_code\PHY_results\Packet', num2str(pktsize(ii)) '_Rate' num2str(Rd) '_iter1000_70mph_SNR_CC_SISO_uplink_rural.mat'];
                    %filename = ['D:\Documents\From Pendrive Digi\Simulator\V2V fizi\Fizi_code\PHY_results_tira'];
                    load('SNR_PHY_spline_div.mat','EbN0'); %load SNR values of PHY lookup table results
                    SNR_dB = EbN0;
                    
                    load('PER_PHY_spline_div.mat','PER'); %load PER for PHY lookup table 
                    
                    
                case 'moving'
                    %filename = ['D:\Documents\From Pendrive Digi\Simulator\V2V fizi\Fizi_code\PHY_results\Packet', num2str(pktsize(ii)) '_Rate' num2str(Rd) '_iter1000_30mph_SNR_CC_SISO_uplink_urban_Msid6_simplestFB.mat'];
                    %filename = '[D:\Documents\From Pendrive Digi\Simulator\V2V fizi\Fizi_code\graph tira'];
                    %load(filename,'SNR_dB','PER');
                    load('SNR_PHY_Linear_WO_Div.mat','EbN0'); %load SNR values of PHY lookup table results
                    SNR_dB = EbN0;
                    load('PER_PHY_Linear_WO_Div.mat','PER'); %load PER for PHY lookup table 
            end
            v =10; % 3km/h,7km/h,10km/h (change the value of speed accordingly)
            %SNR_dB=fliplr(SNR_dB);
            %       CAM/BSM: never fragmented as the content of message is always changing
            %       ACN/DECN: 3 options (not fragmented / fragment at APP / fragment at MAC)
            if pktsize(ii) > 499 % no fragmentation/uncoded
                SNR_acn = SNR_dB;
                PER_acn = PER;
                N_acn = length(PER_acn); % Ps=1-PER
            elseif pktsize(ii) == 300 % no fragmentation/uncoded
                SNR_cam = SNR_dB;
                PER_cam = PER;
                N_cam = length(PER_cam); % Ps=1-PER
            elseif pktsize(ii) < 51 % raptor at MAC
                SNR_MAC = SNR_dB;
                PER_MAC = PER;
                N_MAC = length(PER_MAC);
            else   % raptor at APP
                SNR_APP = SNR_dB;
                PER_APP = PER;
                N_APP = length(PER_APP);
            end
        end
        
        % find PER of specific vehicle
        for kk = 1:SenDensity
            [temp,idx] = min(abs(sensor(k).SNR(kk,t)-SNR_acn));
            sensor(k).PER_acn(kk,t) = PER_acn(idx);
            [temp,idx] = min(abs(sensor(k).SNR(kk,t)-SNR_cam));
            sensor(k).PER_cam(kk,t) = PER_cam(idx);
            [temp,idx] = min(abs(sensor(k).SNR(kk,t)-SNR_MAC));
            sensor(k).PER_MAC(kk,t) = PER_MAC(idx);
            [temp,idx] = min(abs(sensor(k).SNR(kk,t)-SNR_APP));
            sensor(k).PER_APP(kk,t) = PER_APP(idx);
        end
    end %//vehDensity
end %//simTime

toc
%% MAC layer implementation
disp('Block 3: MAC implementation ')
tic
% PRR(pkt received rate), E2E delay and Throughput of ACN (event-triggered) pkts
for k = 1:SenDensity
    k
    dist = sensor(k).dist(:,t);
    pwr = sensor(k).Prx(:,t);
    intf = sensor(k).intf(1:sensor(k).intfnum(1,t),t);
    [x,prio_d] = sort(dist,'ascend'); % interfering vehicles power below threshold
    [y,prio_p] = sort(pwr,'descend'); % interfering vehicles power below threshold
    % No fragmentation @ Uncoded for both ACN and CAM (K=1, PSDU=574B)
    for t = 1:simTime
        for kk = 1:SenDensity
            PER_acn = sensor(k).PER_acn(kk,t);
            [sensor(k).prr_uncoded(kk,t),sensor(k).e2edelay_uncoded(kk,t),sensor(k).Nb_uncoded(kk,t)] = ...
                sps_rssi_mod_new_2(pwr,dist,intf,500,sensor(k).PER_cam(:,t),PER_acn,SenDensityKM,v,1,d_th,ksi,res,d_sense);
            sensor(k).Nr_uncoded(kk,t) = 4; % NEED to check number of repetitions required for FEC do determine Nr ????
        end
    end
    
    for t = 1:simTime
        for kk = 1:SenDensity
            PER_acn = sensor(k).PER_APP(kk,t);
            [sensor(k).prr_APP(kk,t),sensor(k).e2edelay_APP(kk,t),sensor(k).Nb_APP(kk,t)] =...
                sps_rssi_mod_new_2(pwr,dist,intf,200,sensor(k).PER_cam(:,t),PER_acn,SenDensityKM,v,K,d_th,ksi,res,d_sense);
            sensor(k).Nr_APP(kk,t) = floor(sensor(k).Nr_uncoded(kk,t)/sensor(k).e2edelay_APP(kk,t));
        end
    end
    
    for t = 1:simTime
        for kk = 1:SenDensity
            PER_acn = sensor(k).PER_MAC(kk,t);
            [sensor(k).prr_MAC(kk,t),sensor(k).e2edelay_MAC(kk,t),sensor(k).Nb_MAC(kk,t)] =...
                sps_rssi_mod_new_2(pwr,dist,intf,50,sensor(k).PER_cam(:,t),PER_acn,SenDensityKM,v,K,d_th,ksi,res,d_sense);
            sensor(k).Nr_MAC(kk,t) = floor(sensor(k).Nr_uncoded(kk,t)/sensor(k).e2edelay_MAC(kk,t));
        end
    end
end
toc

%% Raptor codes vs. repetition codes implementation
disp('Block 4: FEC schemes implementation (repetition, R10 & RQ codes)')
tic
filename = ['K' num2str(K) '_T512_Table_CR_PER_full.mat'];
load(filename);
%mat=mat_new(1:21,1:19);CR=[1:-0.05:0.1];
CR = [1:-0.02:0.01];mat=mat_full_interp(:,1:length(CR));
filename = 'rep_iter4_K1_Table_prr_overh.mat';
load(filename);
ref = [1:-0.05:0.05,0.01];
per = [1:-0.02:0.01];target=0.01;
for k = 1:SenDensity
    k
    for t = 1:simTime
        for kk = 1:SenDensity
            [x,idxR] = min(abs(sensor(k).prr_uncoded(kk,t)-ref));
            if sensor(k).prr_uncoded(kk,t) < target
                sensor(k).prr_repAPP(kk,t) = 0;
            else
                sensor(k).prr_repAPP(kk,t) = prr_repS(idxR);
            end
            sensor(k).overh_repAPP(kk,t) = overh_repS(idxR);
            sensor(k).e2edelay_repAPP(kk,t) = sensor(k).overh_repAPP(kk,t)*sensor(k).e2edelay_uncoded(kk,t);
            
            [x,idxR] = min(abs(1-sensor(k).prr_APP(kk,t)-per));
            idx = find(mat(idxR,:)<target);
            if ~isempty(idx)
                idxC=idx(1);
                sensor(k).prr_rqAPP(kk,t) = 1;
            else
                idxC = size(mat,2);
                sensor(k).prr_rqAPP(kk,t) = 1-mat(idxR,idxC);
            end
            sensor(k).overh_rqAPP(kk,t) = (K/CR(idxC));
            sensor(k).e2edelay_rqAPP(kk,t) = sensor(k).overh_rqAPP(kk,t)*sensor(k).e2edelay_APP(kk,t);
            
            [x,idxR] = min(abs(1-sensor(k).prr_MAC(kk,t)-per));
            idx = find(mat(idxR,:)<target);
            if ~isempty(idx)
                idxC = idx(1);
                sensor(k).prr_rqMAC(kk,t) = 1;
            else
                idxC = size(mat,2);
                sensor(k).prr_rqMAC(kk,t) = 1-mat(idxR,idxC);
            end
            sensor(k).overh_rqMAC(kk,t) = (K/CR(idxC));
            sensor(k).e2edelay_rqMAC(kk,t) = sensor(k).overh_rqMAC(kk,t)*sensor(k).e2edelay_MAC(kk,t);
        end %//vehDensity
        filename = [phyType,'_results'];
        save(filename);
    end %//simTime
end %% vehDensity
toc



%% plot results
for k = 1:SenDensity
    temp1(k,:) = sensor(k).dist(:);
    temp2r(k,:) = sensor(k).prr_repAPP(:);
    temp3r(k,:) = sensor(k).e2edelay_repAPP(:);
    temp4r(k,:) = sensor(k).overh_repAPP(:);
    
    temp2a(k,:) = sensor(k).prr_rqAPP(:);
    temp3a(k,:) = sensor(k).e2edelay_rqAPP(:);
    temp4a(k,:) = sensor(k).overh_rqAPP(:);
    
    temp2m(k,:) = sensor(k).prr_rqMAC(:);
    temp3m(k,:) = sensor(k).e2edelay_rqMAC(:);
    temp4m(k,:) = sensor(k).overh_rqMAC(:);
    
    temp4(k,:) = sensor(k).Prx(:);
    temp5(k,:) = sensor(k).SNR(:);
    temp6(k,:) = sensor(k).PER_acn(:);
    temp7(k,:) = sensor(k).PER_APP(:);
    temp8(k,:) = sensor(k).PER_MAC(:);
    
    temp9(k,:) = sensor(k).Nb_uncoded(:);
    temp10(k,:) = sensor(k).Nb_APP(:);
    temp11(k,:) = sensor(k).Nb_MAC(:);
end

SNR_all = temp5(:);
dist_all = temp1(:);
prr_repAPP = temp2r(:);
e2edelay_repAPP = temp3r(:);
overh_repAPP = temp4r(:);
prr_rqAPP = temp2a(:);
e2edelay_rqAPP = temp3a(:);
overh_rqAPP = temp4a(:);
prr_rqMAC = temp2m(:);
e2edelay_rqMAC = temp3m(:);
overh_rqMAC = temp4m(:);
Prx_all = temp4(:);
PER_acn_all = temp6(:);
PER_APP_all = temp7(:);
PER_MAC_all = temp8(:);
Nb_acn_all = temp9(:);
Nb_APP_all = temp10(:);
Nb_MAC_all = temp11(:);
switch scen
    case 'stationary'
        par.PL = 'log-distance';
        dist_allN = pathLossModel(par,Ptx,(SNR_all+noiseP),PLexp);
    case 'moving'
        par.PL = 'log-distance';
        dist_allN = pathLossModel(par,Ptx,(SNR_all+noiseP),PLexp);
end

[x idx] = sort(dist_allN,'ascend');
temp = [dist_allN(idx) SNR_all(idx) PER_acn_all(idx) PER_APP_all(idx) PER_MAC_all(idx) prr_repAPP(idx) prr_rqAPP(idx) prr_rqMAC(idx)];
tt1 = idx;
dis = ceil(dist_allN(tt1));
[C,ia,id] = unique(dis,'stable');
dis = C;
prr1 = prr_repAPP(tt1);
val = accumarray(id,prr1,[],@mean);
prr1 = val;
delay1 = e2edelay_repAPP(idx);
val = accumarray(id,delay1,[],@mean);
delay1 = val;
prr2 = prr_rqAPP(tt1);
val = accumarray(id,prr2,[],@mean);
prr2 = val;
delay2 = e2edelay_rqAPP(idx);
val = accumarray(id,delay2,[],@mean);
delay2 = val;
prr3 = prr_rqMAC(tt1);
val = accumarray(id,prr3,[],@mean);
prr3 = val;
delay3 = e2edelay_rqMAC(idx);
val = accumarray(id,delay3,[],@mean);
delay3 = val;
Nb1 = Nb_acn_all(tt1);
val = accumarray(id,Nb1,[],@mean);
Nb1 = val;
Nb2 = Nb_APP_all(tt1);
val = accumarray(id,Nb2,[],@mean);
Nb2 = val;

plot(dis,prr1)
set(0,'DefaultAxesFontName', 'Calibri')
set(0,'DefaultAxesFontWeight', 'Normal')
set(0,'DefaultAxesFontSize', 14)

figure(3);
plot(dis,sort(prr1,'descend'),'-r',dis,sort(prr2,'descend'),'-g',dis,sort(prr3,'descend'),'-b','LineWidth',3);
xlim([0 1000]);
xlabel('distance (m)');
ylabel('Pkt reception rate')
legend('repetition code','RQ code(APP)','RQ code(MAC)');
%legend('repetition code','RQ code(APP)','RQ code(MAC)');grid on;hold on;
grid on;
hold on;

figure(4);
plot(dis,sort(delay1,'ascend'),'-r',dis,sort(delay2,'ascend'),'-g',dis,sort(delay3,'ascend'),'-b','LineWidth',3);
xlim([0 500]);
xlabel('distance (m)');
ylabel('end-to-end delay (ms)');
ylim([0 100])
%legend('repetition code','RQ code(APP)','RQ code(MAC)');grid on;hold on;
legend('repetition code','RQ code(APP)','RQ code(MAC)');
grid on;
hold on;



%end
% figure(1); plot(dist_allN(tt1),prr_repAPP(tt1),'-r',dist_allN(tt1),prr_rqAPP(tt1),'-g',dist_allN(tt1),prr_rqMAC(tt1),'-b','LineWidth',2);
% xlim([0 800]); xlabel('distance (m)'); ylabel('Pkt reception rate')
% legend('repetition code','RQ code (APP)','RQ code (MAC)');grid on;hold on;
% prr=prctile(prr1,0:100);
% plot(prr,0:0.01:1)
%legend('Rep code,\delta=0.25','RQ code,\delta=0.25','Rep code,\delta=0.5','RQ code,\delta=0.5','Rep code,\delta=1','RQ code,\delta=1');grid on;hold on;
