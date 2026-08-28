%function [prr_repS,overh_repS]=fec_repetition_new(prr_unsat,Nd,Nr,K,iterations)
clear all;
prr_unsat=[1:-0.05:0.05,0.01];Nd=length(prr_unsat); Nr=4; K=1; iterations=2000; % repetition at APP

prr_repS = zeros(1,Nd); overh_repS=zeros(1,Nd);   % NUMERICAL
for J = 1:iterations
    ifsucc = zeros(1,Nd);                    
    for k = 1:Nd   
        idx  = (zeros(1,K) > 1);           % index with succesfully received packets - success declared if all are 1  
        for I = 1:Nr % limit to M=Nr tries
            rx  = (rand(1,K) < prr_unsat(k));       % index of successfully received packets    
            idx = (idx | rx);                % elementwise logical or    
        end   
        ifsucc(k) = all(idx);   
    end
    prr_repS = prr_repS + ifsucc;
end
prr_repS = prr_repS/iterations;

    
prr_repT = zeros(1,Nd);
for k = 1:Nd      
    prr_repT(k) =  (1-(1-prr_unsat(k)).^Nr).^K; % THEORY
end

for J=1:iterations    
    for k = 1:Nd
        idx  = (zeros(1,K) > 1); 
        M    = 0;                          % number of retransmissions needed  
        flag = 1;                          % flag: flag = 0 implies success
        while flag
            rx  = (rand(1,K) < prr_unsat(k));       % index of successfully received packets 
            idx = (idx | rx);                % elementwise logical or     
            flag = ~all(idx);
            M = M+1;    
            if M > 1e4                      % infinite loop preventer
                break; 
            end
        end    
        ifsucc(k) = M;   
    end
    overh_repS=overh_repS+ifsucc*K;
end
overh_repS = overh_repS/iterations;

filename=['rep_iter' num2str(Nr) '_K' num2str(K) '_Table_prr_overh.mat']; 
save(filename,'overh_repS','prr_repS')