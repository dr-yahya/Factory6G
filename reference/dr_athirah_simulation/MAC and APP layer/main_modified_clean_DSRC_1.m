% Nor Fadzilah Abdullah
% University of Bristol, UK
% Feb 2017
% Fountain codes for safety broadcast

%% initialisation
clear global; clear all; clc;
rand('state', sum(100*clock));
randn('state', sum(100*clock));

scen='hway'; % 'urban' or 'hway'
phyType='DSRC'; MCS=3; % assumed fixed MCS mode for safety
Ptx=23;% transmit power (dBm)
Gt=3; %dB
Gr=3; %dB
fc=5.9e9; % 802.11p
ant='siso';

%% RaptorQ parameters
SB=512; % source block (bytes)
SS=64; % source symbols (bytes)
K=SB/SS;

%% Mobility model (Cellular Automata: microscopic flow model)
disp('Block 1: Mobility model (CFM)')
tic

tStep=100e-3; % 100ms (period of CAM and ACN repetition as stated in V2V std)
simTime=1;%10 average runs

switch scen
    case 'hway',
        hwayLength=2000; LaneWidth=4; % meter
        numVehPerLaneKM=11; % sparse (2 vehicles/km/lane), moderate (6 vehicles/lane), dense (11 vehicles/lane);
        numLane=6; laneY=[2:LaneWidth:LaneWidth*numLane].'; % 3 lanes/direction
        numVehPerLane=numVehPerLaneKM*hwayLength/1000;
        vehDensityKM=numVehPerLaneKM/1000*numLane;%par.density=[12,36,66]/1000;
        vehDensity=vehDensityKM*hwayLength;
        
        meanInterD=1000/numVehPerLaneKM;
        temp = poissrnd(meanInterD,[numLane,numVehPerLane]);
        vehicle_xLocation=mod(cumsum(temp,2),hwayLength);
        vehicle_xLocation=sort(vehicle_xLocation,2);
        vehicle_yLocation = repmat(laneY,1,numVehPerLane);
        vehicle=struct('xLocation',0,'yLocation',0,'velocity',0);
        
        v_min=30;
        v_max=[60 90 120]; %km/h
        Length=hwayLength;

startVel=[randi([v_min v_max(1)],numLane/2,numVehPerLane); randi([v_max(1) v_max(2)],numLane/2,numVehPerLane)];
startVel=[startVel(2:end,:); startVel(1,:)];

t=1; k=1;
for j=1:numLane
    for i=1:numVehPerLane
        vehicle(k).velocity(t)=startVel(j,i);
        vehicle(k).xLocation(t)=vehicle_xLocation(j,i);
        vehicle(k).yLocation(t)=vehicle_yLocation(j,i);
        k=k+1;
    end
            end
        case 'urban',
            urbanLength=750;urbanWidth=1299;  LaneWidth=3.5;
            numVehPerLaneKM=55 ; % sparse (8 vehicles/km/lane), moderate (16 vehicles/lane), dense (28 vehicles/lane)
            numLane=4*3*2; 
            
            numVehPerLane=(numVehPerLaneKM*urbanWidth/1000);
            vehDensityKM=numVehPerLaneKM/1000*numLane/2;%par.density=[90,198,306]/1000;
            laneX=[2, 5.5, 245, 248.5,252,255.5,495,498.5,502,505.5, 745, 748.5];
            meanInterD=1000/numVehPerLaneKM;
            temp = poissrnd(meanInterD,[numLane/2,numVehPerLane]);
            vehicle_yLocation=[mod(cumsum(temp,2),urbanWidth)];
            vehicle_yLocation1=sort(vehicle_yLocation,2).';
            vehicle_xLocation1 = repmat(laneX,numVehPerLane,1);
            vehicle=struct('xLocation',0,'yLocation',0,'velocity',0);
                        
            numVehPerLane2=(numVehPerLaneKM*urbanLength/1000);
            vehDensityKM2=numVehPerLaneKM/1000*numLane/2;%par.density=[90,198,306]/1000;        
            laneY=[2; 5.5; 428; 431.5;435;438.5;861;864.5;868;871.5; 1294; 1297.5].'; % 2 lanes/direction
            temp = poissrnd(meanInterD,[numLane/2,numVehPerLane2]);
            vehicle_xLocation2=mod(cumsum(temp,2),urbanLength);
            vehicle_xLocation2=sort(vehicle_xLocation2,2);
            vehicle_yLocation2 = repmat(laneY,numVehPerLane2,1).';
                        temp=[7:1:243,257:1:493,507:1:743];

            %vehDensity=vehDensityKM*urbanLength+floor(vehDensityKM*urbanWidth);
            for p=1:numLane/2
                tter(p,:)=ismember(vehicle_xLocation2(p,:),temp);
                
                vehicle_xLocation2(p,:)=tter(p,:).*vehicle_xLocation2(p,:);
                vehicle_yLocation2(p,:)=tter(p,:).*vehicle_yLocation2(p,:);
                
            end
            v_min=10;
            v_max=[15 30 60]; %km/h
            Length=urbanLength;
            Width=urbanWidth;
            
            startVel=[randi([v_min v_max(1)],numLane/2,numVehPerLane); randi([v_max(1) v_max(2)],numLane/2,numVehPerLane)];
            startVel=[startVel(2:end,:); startVel(1,:)];
            startVel2=[randi([v_min v_max(1)],numLane/2,numVehPerLane2); randi([v_max(1) v_max(2)],numLane/2,numVehPerLane2)];
            startVel2=[startVel2(2:end,:); startVel2(1,:)];
            t=1; k=1;
            for j=1:numLane/2
                for i=1:floor(numVehPerLane)
                    vehicle(k).velocity(t)=startVel(j,i);
                    Locations1(k,:)=[vehicle_xLocation1(i,j), vehicle_yLocation1(i,j)];
                    k=k+1;
                end
            end
            for j=1:numLane/2
                for i=1:numVehPerLane2
                    vehicle(k).velocity(t)=startVel2(j,i);
                    Locations1(k,:)=[vehicle_xLocation2(j,i), vehicle_yLocation2(j,i)];
                    k=k+1;
                end
            end
       
[Au,ia,ic] = unique(Locations1, 'rows', 'stable');
RowIdxFreq = accumarray(ic, 1);  
[ss,tt]=find(RowIdxFreq>1);
Locations1(ss,:)=[];
[ss,tt]=find(Locations1<1);
Locations1(ss,:)=[];
k=1;
for loc=1:length(Locations1)
    vehicle(k).xLocation(t)=Locations1(loc,1);
    vehicle(k).yLocation(t)=Locations1(loc,2);
    k=k+1;
end
vehDensity=length(Locations1);
end
switch scen
    case 'hway',
        for t=2:simTime
            t
            for k=1:vehDensity
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
    case 'urban',
        load('Junction_urban.mat');
        for t=2:simTime
            t
            for k=1:vehDensity
                k
                if k<=numVehPerLane || k>(numLane/2)*numVehPerLane
                    vmax_j=v_max(1);
                else
                    vmax_j=v_max(2);
                end
                switch mod(k,numVehPerLane)
                    case 0, j=k/numVehPerLane;
                        precedingVehLoc=[vehicle((j-1)*numVehPerLane+1).xLocation(t-1),vehicle((j-1)*numVehPerLane+1).yLocation(t-1)];
                    otherwise, precedingVehLoc=[vehicle(k+1).xLocation(t-1),vehicle(k+1).yLocation(t-1)];
                end
                dist=sqrt((vehicle(k).xLocation(t-1)-precedingVehLoc(1,1)).^2+(vehicle(k).yLocation(t-1)-precedingVehLoc(1,2)).^2);
                delta_x=abs(dist)/tStep*3.6; % in km/h
                vehicle(k).velocity(t)=min([vmax_j,vehicle(k).velocity(t-1),delta_x]); % accelerate/decelerate
                %Junction_X=[7;243;250;493;500;743]; Junction_Y=[2; 5.5; 428; 431.5;435;438.5;861;864.5;868;871.5; 1294; 1297.5];
                M=length(Junction); Loc=vehicle(k).xLocation(t-1);
                [l p]=find(Loc==Junction(:,1));
                if isempty(p)
                    vehicle(k).xLocation(t)=mod(vehicle(k).xLocation(t-1)+vehicle(k).velocity(t),Length);
                    vehicle(k).yLocation(t)=vehicle(k).yLocation(t-1);
                else
                    TurnRatio=0.25;JTR=[0.25 0.75 1]; % 25% left, 50% straight, 25% right
                    if TurnRatio<=JTR(1)
                        vehicle(k).xLocation(t)=vehicle(k).xLocation(t-1);
                        vehicle(k).yLocation(t)=mod(vehicle(k).yLocation(t-1)-vehicle(k).velocity(t),Width);
                    elseif TurnRatio>=JTR(3)
                        vehicle(k).xLocation(t)=vehicle(k).xLocation(t-1);
                        vehicle(k).yLocation(t)=mod(vehicle(k).yLocation(t-1)+vehicle(k).velocity(t),Width);
                    else
                        vehicle(k).xLocation(t)=mod(vehicle(k).xLocation(t-1)+vehicle(k).velocity(t),Length);
                        vehicle(k).yLocation(t)=vehicle(k).yLocation(t-1);
                    end
                    
                end
            end
        end
end

%% PHY layer Implementation
disp('Block 2: PHY implementation (SNR vs. PER)')
tic
% IEEE 802.11p MCS peak data rates
Table = [3    1/2 1 48  24;     % BPSK 1/2 (Rd, Rc, N_c, N_s, N_d)
    4.5  3/4 1 48  36;     % BPSK 3/4
    6    1/2 2 96  48;     % QPSK 1/2 % *** choose this
    9    3/4 2 96  72;     % QPSK 3/4
    12   1/2 4 192 96;     % 16QAM 1/2
    18   3/4 4 192 144;    % 16QAM 3/4
    24   2/3 6 288 192;    % 64QAM 2/3
    27   3/4 6 288 216];   % 64QAM 3/4

Rd=Table(MCS,1);
Rc=Table(MCS,2);
N_c=Table(MCS,3);               % Coded_bit_symbol
N_s=Table(MCS,4);               % Coded_bits
N_d=Table(MCS,5);               % Coded_data
B=10e6; % BW for IEEE 802.11p
NF=7; % noise figure
RxTh=[-85 -84 -89.6 -80 -77 -73 -69 -68]; % RX sensitivity (refer std)
%acheive per=0.01;
RxSen=RxTh(MCS);
% noise power
% boltzmannK=1.3806485279e-23;
thermalNoise =-174; % = 10*log10(boltzmannK*290K*1000) dBM/Hz;
noiseP=thermalNoise+10*log10(B)+NF; % in dBm
IM=5;% implementation loss;
lambda= 3e8/fc;
switch scen
    case 'hway', snr=5.2;RxSen=-89.8;PLexp=2.7;d_th=150;
    case 'urban', d_th=320; snr=5.4;RxSen=-89.6;PLexp=3.1;% snr should around 5.4+6=11.4 check phy simulations
end
% discrete-time simulation
for t=1:simTime % translate dist->SNR and find out FEC performance
    for k=1:vehDensity
        %         vehicle(k).intf=zeros(vehDensity,simTime);
        % find SNR
        for kk=1:vehDensity
            vehicle(k).dist(kk,t)= sqrt( (vehicle(k).xLocation(t)-vehicle(kk).xLocation(t))^2 + (vehicle(k).yLocation(t)-vehicle(kk).yLocation(t))^2);
            if vehicle(k).dist(kk,t)==0
                vehicle(k).dist(kk,t)=0.001;
            end
            vehicle(k).Prx(kk,t)= Ptx+Gt+Gr+20*log10(lambda/4/pi)+10*log10((1/vehicle(k).dist(kk,t))^PLexp);
        end
        temp=find(~isinf(vehicle(k).Prx(:,t))&vehicle(k).Prx(:,t)>RxSen);
        %idx=(rand(length(temp),1) < 0.1 ); temp = temp(idx); %(10% simulataneous V2V tx)
        vehicle(k).intf(1:numel(temp),t)=temp;
        vehicle(k).intfnum(1,t)=numel(temp);
        for kk=1:vehDensity
            vehicle(k).SNR(kk,t)= vehicle(k).Prx(kk,t) - 10*log10(10^(noiseP/10) + 10^(sum(vehicle(k).Prx(temp,t))/10)); % SINR
        end

        pktsize=[105 159 337 837]; midsym=10;
        for ii=1:length(pktsize)
            switch scen
                case 'hway',
                    filename=['Y:\Year3\V2V_Simulator_Safety_Fizi\PHY_results\IEEE80211p_SISO_phy_Rd3_', num2str(pktsize(ii)) 'B_70mph_preamble_rural.mat'];
                    load(filename);
                case 'urban'
                    filename=['Y:\Year3\V2V_Simulator_Safety_Fizi\PHY_results\IEEE80211p_SISO_phy_Rd3_', num2str(pktsize(ii)) 'B_30mph_preamble_urban_Msid6.mat'];
                    load(filename);
            end
            v=33; % in this case, fixed speed v=28m/s=100km/h=62.5mph          
            %       CAM/BSM: never fragmented as the content of message is always changing
            %       ACN/DECN: 3 options (not fragmented / fragment at APP / fragment at MAC)
            if pktsize(ii)==837 % no fragmentation/uncoded
                SNR_acn=SNR_dB; PER_acn=PER; N_acn=length(PER_acn); % Ps=1-PER
            elseif pktsize(ii)==337 % no fragmentation/uncoded
                SNR_cam=SNR_dB; PER_cam=PER; N_cam=length(PER_cam); % Ps=1-PER
            elseif pktsize(ii)==105 % raptor at MAC
                SNR_MAC=SNR_dB; PER_MAC=PER; N_MAC=length(PER_MAC);
            else % pktsize(ii)=126, raptor at APP
                SNR_APP=SNR_dB; PER_APP=PER; N_APP=length(PER_APP);
            end
        end
        % find PER of specific vehicle
        for kk=1:vehDensity
            [temp,idx]=min(abs(vehicle(k).SNR(kk,t)-SNR_acn));
            vehicle(k).PER_acn(kk,t)=PER_acn(idx);
            [temp,idx]=min(abs(vehicle(k).SNR(kk,t)-SNR_cam));% 5dB per=0.01
            vehicle(k).PER_cam(kk,t)=PER_cam(idx);
            [temp,idx]=min(abs(vehicle(k).SNR(kk,t)-SNR_MAC));
            vehicle(k).PER_MAC(kk,t)=PER_MAC(idx);
            [temp,idx]=min(abs(vehicle(k).SNR(kk,t)-SNR_APP));
            vehicle(k).PER_APP(kk,t)=PER_APP(idx);
        end
    end %//vehDensity
end %//simTime

toc
%% MAC layer implementation
disp('Block 3: MAC implementation ')
tic
% PRR(pkt received rate), E2E delay and Throughput of ACN (event-triggered) pkts
for k=1:vehDensity
    % No fragmentation @ Uncoded for both ACN and CAM (K=1, PSDU=574B)
    midsym=10;
    for t=1:simTime
        for kk=1:vehDensity
            PER_acn=vehicle(k).PER_acn(kk,t);
            [vehicle(k).prr_uncoded(kk,t),vehicle(k).e2edelay_uncoded(kk,t),vehicle(k).Tput_uncoded(kk,t)]=...
                csma_sbrod_unsat_edca_allCodes_v3(Rd,N_d,d_th,'IFS','APP','repetition',1,PER_acn,vehicle(k).PER_cam(kk,t),v,vehDensityKM,midsym,scen);
            vehicle(k).Nr_uncoded(kk,t)=4; % NEED to check number of repetitions required for FEC do determine Nr ????
        end        
    end
    % ACN Fragmentation at APP (K=8, PSDU=126B)
    for t=1:simTime
        for kk=1:vehDensity
            PER_acn=vehicle(k).PER_APP(kk,t);
            [vehicle(k).prr_APP(kk,t),vehicle(k).e2edelay_APP(kk,t),vehicle(k).Tput_APP(kk,t)]=...
                csma_sbrod_unsat_edca_allCodes_v3(Rd,N_d,d_th,'IFS','APP','raptor',K,PER_acn,vehicle(k).PER_cam(kk,t),v,vehDensityKM,midsym,scen);
            vehicle(k).Nr_APP(kk,t)=floor(vehicle(k).Nr_uncoded(kk,t)/vehicle(k).e2edelay_APP(kk,t));
        end
    end
    % ACN Fragmentation at MAC (K=8, PSDU=72B)
    for t=1:simTime
        for kk=1:vehDensity
            PER_acn=vehicle(k).PER_MAC(kk,t);
            [vehicle(k).prr_MAC(kk,t),vehicle(k).e2edelay_MAC(kk,t),vehicle(k).Tput_MAC(kk,t)]=...
                csma_sbrod_unsat_edca_allCodes_v3(Rd,N_d,d_th,'IFS','MAC','raptor',K,PER_acn,vehicle(k).PER_cam(kk,t),v,vehDensityKM,midsym,scen);
            vehicle(k).Nr_MAC(kk,t)=floor(vehicle(k).Nr_uncoded(kk,t)/vehicle(k).e2edelay_MAC(kk,t));
        end
    end    
end
toc

%% Raptor codes vs. repetition codes implementation
disp('Block 4: FEC schemes implementation (repetition, R10 & RQ codes)')
tic
filename=['K' num2str(K) '_T512_Table_CR_PER_full.mat'];
load(filename);
%mat=mat_new(1:21,1:19);CR=[1:-0.05:0.1];
CR=[1:-0.02:0.01];mat=mat_full_interp(:,1:length(CR));
filename='rep_iter4_K1_Table_prr_overh.mat';
load(filename);
ref=[1:-0.05:0.05,0.01];
per=[1:-0.02:0.01];target=0.01;
for k=1:vehDensity
    k
    for t=1:simTime
        for kk=1:vehDensity
            [x,idxR]=min(abs(vehicle(k).prr_uncoded(kk,t)-ref));
            if vehicle(k).prr_uncoded(kk,t)<target
                vehicle(k).prr_repAPP(kk,t)=0;
            else
                vehicle(k).prr_repAPP(kk,t)=prr_repS(idxR);
            end
            vehicle(k).overh_repAPP(kk,t)=overh_repS(idxR);
            vehicle(k).e2edelay_repAPP(kk,t) = vehicle(k).overh_repAPP(kk,t)*vehicle(k).e2edelay_uncoded(kk,t);

                [x,idxR]=min(abs(1-vehicle(k).prr_APP(kk,t)-per));
                idx=find(mat(idxR,:)<target);
                if ~isempty(idx)
                idxC=idx(1);
                vehicle(k).prr_rqAPP(kk,t)=1;
                else
                idxC=size(mat,2);
                vehicle(k).prr_rqAPP(kk,t)=1-mat(idxR,idxC);
                end
            vehicle(k).overh_rqAPP(kk,t)=(K/CR(idxC));
            vehicle(k).e2edelay_rqAPP(kk,t) = vehicle(k).overh_rqAPP(kk,t)*vehicle(k).e2edelay_APP(kk,t);

                [x,idxR]=min(abs(1-vehicle(k).prr_MAC(kk,t)-per));     
                idx=find(mat(idxR,:)<target);
                if ~isempty(idx)
                idxC=idx(1);
                vehicle(k).prr_rqMAC(kk,t)=1;
                else
                idxC=size(mat,2);
                vehicle(k).prr_rqMAC(kk,t)=1-mat(idxR,idxC);
                end
            vehicle(k).overh_rqMAC(kk,t)=(K/CR(idxC));
            vehicle(k).e2edelay_rqMAC(kk,t) = vehicle(k).overh_rqMAC(kk,t)*vehicle(k).e2edelay_MAC(kk,t);
        end %//vehDensity
        filename=[phyType,'_results'];
        save(filename);
    end %//simTime
end %% vehDensity
toc
%% plot results
for k=1:vehDensity
    temp1(k,:)=vehicle(k).dist(:);
    temp2r(k,:)=vehicle(k).prr_repAPP(:);
    temp3r(k,:)=vehicle(k).e2edelay_repAPP(:);
    temp4r(k,:)=vehicle(k).overh_repAPP(:);
    
    temp2a(k,:)=vehicle(k).prr_rqAPP(:);
    temp3a(k,:)=vehicle(k).e2edelay_rqAPP(:);
    temp4a(k,:)=vehicle(k).overh_rqAPP(:);
    
    temp2m(k,:)=vehicle(k).prr_rqMAC(:);
    temp3m(k,:)=vehicle(k).e2edelay_rqMAC(:);
    temp4m(k,:)=vehicle(k).overh_rqMAC(:);
    
    temp4(k,:)=vehicle(k).Prx(:);
    temp5(k,:)=vehicle(k).SNR(:);
    temp6(k,:)=vehicle(k).PER_acn(:);
    temp7(k,:)=vehicle(k).PER_APP(:);
    temp8(k,:)=vehicle(k).PER_MAC(:);
end

SNR_all=temp5(:);
dist_all=temp1(:);
prr_repAPP=temp2r(:); e2edelay_repAPP=temp3r(:); overh_repAPP=temp4r(:);
prr_rqAPP=temp2a(:); e2edelay_rqAPP=temp3a(:); overh_rqAPP=temp4a(:);
prr_rqMAC=temp2m(:);  e2edelay_rqMAC=temp3m(:); overh_rqMAC=temp4m(:);
Prx_all=temp4(:);
PER_acn_all=temp6(:);
PER_APP_all=temp7(:);
PER_MAC_all=temp8(:);

switch scen
    case 'hway',
        par.PL='log-distance'; dist_allN=pathLossModel(par,Ptx,(SNR_all+noiseP),PLexp);
    case 'urban'
        par.PL='log-distance'; dist_allN=pathLossModel(par,Ptx,(SNR_all+noiseP),PLexp);
end

[x idx]=sort(dist_allN,'ascend');
temp=[dist_allN(idx) SNR_all(idx) PER_acn_all(idx) PER_APP_all(idx) PER_MAC_all(idx) prr_repAPP(idx) prr_rqAPP(idx) prr_rqMAC(idx)];
set(0,'DefaultAxesFontName', 'Calibri')
set(0,'DefaultAxesFontWeight', 'Normal')
set(0,'DefaultAxesFontSize', 14)
figure(3); plot(dist_allN(idx),prr_repAPP(idx),'--r',dist_allN(idx),prr_rqAPP(idx),'--g',dist_allN(idx),prr_rqMAC(idx),'--b','LineWidth',3);
xlim([0 800]); xlabel('distance (m)'); ylabel('Average PRR (Packet Reception Rate)')
legend('IEEE802.11p-repetition code','IEEE802.11p-RQ code(APP)','IEEE802.11p-RQ code(MAC)');grid on;hold on;

figure(4); plot(dist_allN(idx),e2edelay_repAPP(idx)*1e3,'--r',dist_allN(idx),e2edelay_rqAPP(idx)*1e3,'--g',dist_allN(idx),e2edelay_rqMAC(idx)*1e3,'--b','LineWidth',3);
xlim([0 800]); xlabel('distance (m)'); ylabel('end-to-end delay (ms)'); ylim([0 20])
legend('IEEE802.11p-repetition code','IEEE802.11p-RQ code(APP)','IEEE802.11p-RQ code(MAC)');grid on;hold on;

legend('C-V2V-repetition code','C-V2V-RQ code(APP)','C-V2V-RQ code(MAC)','IEEE802.11p-repetition code','IEEE802.11p-RQ code(APP)','IEEE802.11p-RQ code(MAC)');grid on;hold on;
