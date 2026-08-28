function [prr,Edn,S_acn,T,PktCollect,prr_cam,Edn_cam]=csma_sbrod_unsat_edca_allCodes_v3(Rd,N_d,R,MACtype,layer,Code,K,Pe,Pe_cam,v,vd,midsym,scen)

tStep=1;
Rd=Rd*1e6;
SIFS=32; DIFS=58; % us
ts=13; % slot time (us)
tp=1; % propagation delay (us)
W0=31; % minimum contention window (31)
T_ofdm=8; % 1 OFDM=8us
overh=1;
Per=0;
 
%% wait interframe symbol before next pkt transmission (sense channel) or not
switch MACtype
    case 'IFS' % sent 1 ACN pkt at a time
        aifsn=2; % ACN
        AIFS = SIFS+aifsn*ts;
        aifsn=9; % CAM
        AIFS_cam = SIFS+aifsn*ts;
    case 'Stream' % stream (aggregate ACN pkts)
        AIFS=0; % ACN
        aifsn=9; % CAM
        AIFS_cam = SIFS+aifsn*ts;
end

%% emergency pkt (ACN)
if strcmp(Code,'repetition')
    K=1;
    payload=800-25;
    %Pe=Pe_cam;
elseif strcmp(Code,'raptor') && strcmp(layer,'APP') 
    payload=round((800-25)/K); % 97B
else     % ('raptor','MAC')
    payload=837/K; %105B
end

if strcmp(layer,'APP')
    pktsize = ceil((payload + 8+20 + 34 )*8/N_d) ; % payload + udp+ip+ mac % + phy headers (#sym)
else %if strcmp(layer,'MAC')
    pktsize = ceil(payload*8/N_d) ; %
end
midamble = 2*ceil(pktsize/midsym); % 2 OFDM symbols for each midamble (SISO & STBC2x2)
% if overh==1 % analytical
P = pktsize*T_ofdm; 

%H = 5*T_ofdm+midamble*T_ofdm; %us (40= T_ofdm*(2*L_STF+2*L_LTF+L_SIG))
H = 5*T_ofdm;
W = (W0+1)/4-1; 
tao_acn=2/(W+1); % probability that a vehicle transmits ACN packet (saturation condition, p0=0)
pgi=1/(K*10); % Packet generation interval (s)
lamda=ts/(pgi)/1e+6; % norm PGR (pkt/s) 

Tb = (P+H); % channel sensed busy by each node in the network (us) /Rd*1e6
% Tc Ts computation for basic access
T = Tb + AIFS + ts + tp; % backoff timer suspended and deferred 
Tss=floor(Tb/ts);
Tcc=floor(T/ts);

%% status beacon (CAM) : do not fragment because content always changing 
payload_app_cam=300-25;
P_cam=ceil((payload_app_cam + 8+20 + 34 )*8/N_d)*T_ofdm; 
H_cam=H; 
W_cam = W0;
tao_cam=2/(W+W_cam+1); % probability that a vehicle transmits CAM packet (saturation condition, p0=0)
pgi_cam=1/10; 
lamda_cam=ts/(pgi_cam)/1e+6;

Tb_cam = (P_cam+H_cam); % channel sensed busy by each node in the network (us) /Rd*1e6
T_cam = Tb_cam + AIFS_cam + ts + tp; % backoff timer suspended and deferred

Tss_cam=floor(Tb_cam/ts);
Tcc_cam=floor(T_cam/ts);

%% paper: Performance and Reliability of DSRC Vehicular Safety Communication: A Formal Analysis
% R = Communication range 
Lcs = R; % assume sensing range = transmission range

if overh==1 %& strcmp(scen,'highway') % analytical highway 
    beita=vd/1e3; % vd (vehicles/km -> vehicles/m)
    if overh==1 && strcmp(scen,'urban') % analytical urban
    beita=beita/1e3*2;
    R=R^2;
    Lcs=Lcs^2;
    end
    Ntr = 2*beita*R;  
    Ncs = 2*beita*Lcs; 
    Nph = 4*beita*R-Ntr; % #potential hidden nodes
else % numerical
    Ntr=vd; Ncs=Ntr; Nph=Ntr;
    beita=Ntr/2/R;    
end

%% service time computation
p0=0; p1=0; di=1; p0_cam=0; p1_cam=0; di_cam=1; % initialize as saturation condition

loop=1;
while di>1e-5 % p0 & p1 do not converge
    p0=p1; p0_cam=p1_cam;   
    
    tao=tao_acn+tao_cam;
    Pb=1-exp(-2*beita*Lcs*((1-p0)*tao)); % probability channel is sensed busy by the tagged vehicle (ACN)
    
    % Service time distribution for ACN (emergency) pkt
    step=1e-8;
    z=1:step:1+step;
    Hd=(1-Pb)*z+Pb*(z.^Tcc);
    sum_acn=zeros(1,2);
    for j=1:2
        for i=1:W
            sum_acn(j)=sum_acn(j)+Hd(j)^(i-1);
        end
    end
    Q=z.^Tss.*sum_acn./W;
    der(1)=diff(Q)/diff(z); % 1st order differentiation computation
    mus=1/der(1); % service rate
    ser=der*ts; % average service time 

    % Service time distribution for CAM (beacon) pkt
    z=1:step:1+step;
    Hd_cam=(1-Pb)*z+Pb*(z.^Tcc_cam);
    sum_cam=zeros(1,2);
    for j=1:2
        for i=W:W_cam-1
            sum_cam(j)=sum_cam(j)+Hd_cam(j)^i;
        end
    end
    Q_cam=z.^Tss_cam.*sum_cam./(W_cam-W);
    der_cam(1)=diff(Q_cam)/diff(z); %differential equation
    mus_cam=1/der_cam(1);
    ser_cam=der_cam*ts;

    rho=(lamda+lamda_cam)/(mus+mus_cam); % arrival rate/service rate
    if rho <=1 % non-saturation condition
        p1=1-lamda/(mus+mus_cam);
        p1_cam=1-lamda_cam/(mus+mus_cam);
    else % saturation condition
        p1=0; p1_cam=0;
    end
    di=abs(p1-p0);
    loop=loop+1;
end    

%% 2nd order differentiation computation

% ACN (emergency) pkt
z=1+step:step:1+2*step;
Hd=(1-Pb)*z+Pb*(z.^Tcc);
sum_acn=zeros(1,2);
for j=1:2
    for i=1:W
        sum_acn(j)=sum_acn(j)+Hd(j)^(i-1);
    end
end
Q=z.^Tss.*sum_acn./W;
der(2)=diff(Q)/diff(z); % 2nd order differentiation computation
tao_acn=(1-p0)*tao_acn;

% CAM (beacon) pkt
z=1+step:step:1+2*step;
Hd_cam=(1-Pb)*z+Pb*(z.^Tcc_cam);
sum_cam=zeros(1,2);
for j=1:2
    for i=W:W_cam-1
        sum_cam(j)=sum_cam(j)+Hd_cam(j)^i;
    end
end
Q_cam=z.^Tss_cam.*sum_cam./(W_cam-W);
der_cam(2)=diff(Q_cam)/diff(z); %differential equation
tao_cam=(1-p0_cam)*tao_cam;

%% delay (Edn), throughput (S) and pkt reception rate (PRR) computation

tao=(tao_acn+tao_cam); 

% Probabilities as a result of sequential backoff process (for saturation condition only)
Ptr=1-exp(-Ncs*tao); 

% hidden vulnerable period, normalised to Rd (refer to cam or acn?)
Tvuln=2*(P+H); 
Tvuln_cam=2*(P_cam+H_cam); 

% link breaking probability for a communication pair
Plb=1-exp(-beita*v*T/1e6); 
Plb_cam=1-exp(-beita*v*T_cam/1e6);

% probability transmission from the tagged node (ACN) is successful
Ps=tao_acn*exp(-(Ncs+(Tvuln/(ts+Pb*T))*Nph-1)*tao)*(1-Pe)*(1-Plb)^Ntr; 
Ps_cam=tao_cam*exp(-(Ncs+(Tvuln_cam/(ts+Pb*T_cam))*Nph-1)*tao)*(1-Pe_cam)*(1-Plb_cam)^Ntr; 

% probability of a collision seen by a packet being transmitted in the medium ...
% @ probability of at least one collision in the medium among other vehicles in the interference range of the tagged vehicle under consideration
Pc=1-exp(-tao*(Ncs+(Tvuln/(ts+Pb*T))*Nph)); 
Pc_cam=1-exp(-tao*(Ncs+(Tvuln_cam/(ts+Pb*T_cam))*Nph)); 

% throughput calculation
if rho<1 % non-saturation (Pollaczek-Khintchine mean value formula)
    Eq=lamda*((diff(der)/diff(z)+der(1)))/(2*(1-lamda/(mus+mus_cam))); % (1-lamda/(mus+mus_cam))=p0
    Edn=(Eq+ser+AIFS+ts+tp)*1e-6;         %AIFS+ts+tp  vs. T
    Eq_cam=lamda_cam*(diff(der_cam)/diff(z)+der_cam(1))/(2*(1-lamda_cam/(mus+mus_cam)));
    Edn_cam=(Eq_cam+ser_cam+AIFS_cam+ts+tp)*1e-6;    %AIFS_cam+ts+tp vs. T_cam
    
    S_acn=Ntr*(lamda+lamda_cam)*P*(1-Pc); 
    S_cam=Ntr*(lamda+lamda_cam)*P_cam*(1-Pc_cam);      
else % saturation
    Edn=(W+1)/2*((1-Ptr)*ts+(Ptr)*Ps*T+(Ptr)*Pc*T+(Ptr)*Per*T)*1e-6;
    Edn_cam=(W_cam+1)/2 *( (1-Ptr_cam)*ts + Ptr_cam*Ps_cam*T_cam + Ptr_cam*Pc_cam*T_cam + Ptr_cam*Per_cam*T_cam )*1e-6;
    
    S_acn=Ntr*Ps*P/((1-Pb)*ts + Pb*T);
    S_cam=Ntr*Ps_cam*P_cam/((1-Pb)*ts + Pb*T_cam);
end

% constant
C = beita*Tvuln*tao/((1-Pb)*ts + Pb*T); %acn=448, cam=1648
C_cam = beita*Tvuln_cam*tao/((1-Pb)*ts + Pb*T_cam); 

prr1=1/(R*C)*(1-exp(-R*C)); % ratio of receivers free from collisions caused by hidden nodes
prr2=exp(-beita*R*tao); % ratio of successful receiving nodes in the range [0,R]
prr3=1/(beita*R*tao)*(1-exp(-beita*R*tao)); % ratio of receivers in [0,R] free from collisions caused by concurrent transmissions of nodes in the range [-R, 0]
prrf=prr1*prr2*prr3*(1-Pe)*(1-Plb)^Ntr; % PRR for a single packet transmission @ first packet in multiple packet transmissions
prr=prrf;

prr1_cam=1/(R*C_cam)*(1-exp(-R*C_cam)); % ratio of receivers free from collisions caused by hidden nodes
prrf_cam=prr1_cam*prr2*prr3*(1-Pe_cam)*(1-Plb_cam)^Ntr; % PRR for a single packet transmission @ first packet in multiple packet transmissions
prr_cam=prrf_cam;

% Edn1=Edn;
Edn=Edn*overh;

PktCollect = tStep/Edn; %= overh/Edn1*tStep;
if Pe==1 & overh~=1
    PktCollect=0;
    S_acn=0;
    Edn=1e4;
end



