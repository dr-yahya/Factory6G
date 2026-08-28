function [prr,macdelay,max_rx]=sps_rssi_mod_new_2(pwr,dist,intf,psize,pe_cam,pe_acn,vd,v,K,d_th,ksi,res,d_sense)
    % For Periodic traffic, working assumption of message size is that one 300-byte message followed by four 190-byte messages, and the time instance of 300-byte size message generation is randomized among vehicles. Note that it is allowed not to consider message size in calculating the performance metric.
    % For Event-triggered traffic, event arrival follows Poisson process with the arrival rate X (up to company choice) per second for each vehicle. Once event triggered, 6 messages are generated with space of 100ms. Working assumption of message size for Event-trigger traffic at L1 is 800bytes.
    SC_RB=4*res;%1Subchannel=4 data RB pairs
     Num_RB_DENM=ceil((psize*8)/(108)); % 12*9=108bits; 12 subcarriers and 9 slots used for data
    Num_RB_CAM=ceil((8*300)/(108));
   Num_SC_DENM=ceil(Num_RB_DENM/SC_RB); %
    Num_SC_CAM=ceil(Num_RB_CAM/SC_RB);
    CAM_time=(Num_SC_CAM*0.1); % 1ms is divided in to 10 subchannels with 5 RB/subchannel. 20%*5RB=4RB used for data
    DENM_time=(Num_SC_DENM*0.1); % DENM has high priority

    %%%%%%%%%%%%%%%%%
    PRB_data=40*res;
%     CAM_time=(Num_RB_CAM/PRB_data);
%     DENM_time=(Num_RB_DENM/PRB_data);
    if ksi==0
        Nb_CAM=floor(100/ceil(Num_RB_CAM/PRB_data));
        Nb_DENM=floor(100/ceil(Num_RB_DENM/PRB_data));
    else
        Nb_CAM=10*floor((PRB_data*10)/Num_RB_CAM);
        Nb_DENM=10*floor((PRB_data*10)/Num_RB_DENM);
    end
    
    %beita=vd/1e3; % vd (vehicles/km -> vehicles/m)
    [x,prio]=find(dist(intf)<d_sense);
    Tx=numel(prio);
    %% added max traffic conditions
        %Rx=0.2*320;
    %%
    if ksi==0
        CAM_total_time=Tx*ceil(CAM_time);
        DENM_total_time=ceil(DENM_time);
        max_rx_time=floor(100-(DENM_total_time+CAM_total_time));
        max_rx=floor(max_rx_time/ceil(DENM_time));
    else
        CAM_total_time=Tx*(CAM_time);
        max_rx_time=floor(100-(DENM_time+CAM_total_time));
        max_rx=floor(max_rx_time/DENM_time);
    end
    [p,rx]=find(dist(:)<d_th); %d_th=1400(rural),700(urban)
    Rx=numel(rx);
    if max_rx>Rx
        tao_acn=1;
    elseif max_rx<0
        tao_acn=0;
    else
        %P_out=(1-pe_cam(dist<=R));X=numel(P_out)*(max_rx/Y);
        tao_acn=(max_rx/Rx);
    end
    %     [x,prio]=find(dist(intf)<d_th);
    % Tx=numel(prio);
    % max_rx=100-DENM_time-Tx*CAM_time;
    % if max_rx>Tx
    %     tao_acn=1;
    % else
    %     tao_acn=max_rx/Tx;
    % end
    % Ntr = (beita*R); %average number of vehicles within 320m
    % prr1=1/(R*C)*(1-exp(-R*C)); % ratio of receivers free from collisions caused by hidden nodes
    %     prr2=exp(-beita*R*tao_acn); % ratio of successful receiving nodes in the range [0,R]
    %     prr3=1/(beita*R*tao_acn)*(1-exp(-beita*R*tao_acn)); % ratio of receivers in [0,R] free from collisions caused by concurrent transmissions of nodes in the range [-R, 0]
    %     %prr=prr2*(1-pe_acn); % PRR for a single packet transmission @ first packet in multiple packet transmissions
    %     %prr=prr2*prr3*(1-pe_acn)*(1-Plb)^Ntr; % PRR for a single packet transmission @ first packet in multiple packet transmissions
    %     % link breaking probability for a communication pair
    %     Plb=1-exp(-beita*v*CAM_time/1e3);
    %     Y=ceil(2*R*vd);
    %prr=prr2*(1-pe_acn)*(1-Plb)^Ntr;
    prr=tao_acn*(1-pe_acn);
    macdelay=DENM_time;%ms
    %only non-saturation condition considered where arrival rate=service
    %rate
    %%%%%
    
    
    
    