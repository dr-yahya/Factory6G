%Athirah Mohd Ramly
%National University of Malaysia (UKM)
%athirah.ramly90@gmail.com
%Joint Iterative Decoding and Detection using polar coded SCMA-OFDM with
%channel estimation at PHY layer
%% Initilization
tic
EbN0 = 0:1:10; 
polar_N = 256;
polar_K = 128;
polar_n = log2(polar_N);
construction_method = 0;%0-BA,1-MC,2-GA
design_snr_dB = 0;%BA and MC construction method will use it
sigma = 0.9;%GA construction method will use it
crc_size = 0;
global pilot_loc Xp
[FZlookup,bitreversedindices,F_kron_n] = initPC(polar_N,polar_K,polar_n,construction_method,design_snr_dB,sigma,crc_size);
alpha = 0.6;
iter_num = 5;
isInterleaver = 1;

%% load codebook
load('codebook_6users_4chips_qpsk.mat','CB');
%load('channel_custom_complex.mat','channel');

K = size(CB, 1); % number of orthogonal resources
M = size(CB, 2); % number of codewords in each codebook
V = size(CB, 3); % number of users (layers)
%polar initial and encoding



SCAN_ITER_NUM = 1;
N = polar_N/log2(M); %Number of scma symbols of each user
SNR  = EbN0 + 10*log10(polar_K/polar_N*log2(M)*V/K);
N0 = 1./10.^(SNR/10); % Noise variance

Nerr = zeros(1,length(EbN0));
Nbits = zeros(1,length(EbN0));
BER   = zeros(1, length(EbN0));

%maxNumErrs = 10000;
maxNumBits = 1e7; %total numer of bits
minNumBits = 50000;
minNumErrs = 50;


% maxNumPEs = 10; % The maximum number of packet errors at an SNR point
% maxNumPackets = 100; % Maximum number of packets at an SNR point
% S = numel(SNR);
% packetErrorRate = zeros(S,1);
% 
% 
% % Loop to simulate multiple packets
% numPacketErrors = 0;
% numPkt = 1; % Index of packet transmitted

%% polar encoder
    for iter_ebn0 = 1:length(EbN0)
        
        
        while ((min(Nerr(:,iter_ebn0)) < minNumErrs) && (Nbits(1,iter_ebn0) < maxNumBits) || (Nbits(1,iter_ebn0) <minNumBits) )%100 010 000
            
            
   %% generate bits         
            infobits = randi([0 1],V,polar_K);
            c = zeros(V,polar_N);
            for user = 1:V
                c(user,:) = pencode(infobits(user,:),FZlookup,crc_size,bitreversedindices,F_kron_n);
            end
   %% interleaver         
            if isInterleaver ~= 0
                interleaver = zeros(V,polar_N);
                interleavered_bits = zeros(size(c));
                for ii = 1:V
                    interleaver(ii,:) = randperm(polar_N);
                    interleavered_bits(ii,:) = c(ii, interleaver(ii,:));
                end
                
            else
                interleavered_bits = c;
            end
            
            temp1 = reshape(interleavered_bits',polar_N*V,1);
            temp2 = reshape(temp1,log2(M),N*V);
            x_temp = bi2de(temp2',log2(M),'left-msb');
            x = reshape(x_temp,N,V);
            x = x';
            %% Channel 
            %h = channel;
            %h = 1/sqrt(2)*(repmat(randn(1, V, N), K, 1)+1j*repmat(randn(1, V, N), K, 1)); %UL Rayleigh W/o Diversity
            %h = 1/sqrt(2)*(randn(K, V, N)+1j*randn(K, V, N)); % Rayleigh channel
            h = ones(K, V, N); % perfect CSI
            %h = 1/sqrt(2)*(repmat(randn(1, V, N), K, 1)+1j*repmat(randn(1, V, N), K, 1));
            %h = 1/sqrt(2)*(repmat(randn(K, 1, N), 1, V)+1j*repmat(randn(K, 1, N), 1, V)); % DL with Diversity
            %h = 1/sqrt(2)*(repmat(repmat(randn(1, 1, N),K, 1), 1,...
            %V)+1j*(repmat(repmat(randn(1, 1, N), K, 1), 1, V))); %DL Rayleigh
            %w/o Diversity
            
            s = scmaenc(x, CB, h);
            %% Add Pilot
            
            s_pilot = add_pilot(s);
            
            %% IFFT
            
            s_pilot_ifft = ifft(s_pilot);
            
            %% Add Cyclic Prefix
            
            s_pilot_ifft_cp = add_CP(s_pilot_ifft);
            
            
            %% parallel to serial
            
            S_pilot_ifft_cp_serial = s_pilot_ifft_cp.';
            
            %% AWGN
            
            y = awgn(S_pilot_ifft_cp_serial, SNR(iter_ebn0),'measured');
            
            %% Serial to parallel
            
            y_parallel = y.';
            %% Remove Cyclic Prefix
            
            y_pilot_ifft = remove_CP(y_parallel);
            
            %% FFT
            
            y_pilot = fft(y_pilot_ifft);
                        for snr = 0:2:30
            %% Channel estimation
            
            %Least Square
            
            Nfft=640;
            Nps=4;
            %H_est = LS_CE(y_pilot,Xp,pilot_loc,Nfft,Nps,'linear');
            %                 %err=(H-H_est)*(H-H_est)';
            %                 %z=[z err/(Nfft*Nsym)];
            %                 %y_eq = LS_CE2(y);
            
            %MMSE
            
                            H_est = MMSE_CE(y_pilot,Xp,pilot_loc,Nfft,Nps,h,snr);
                        end
            %% Equalization
            
            y_eq = y_pilot./H_est;
            
            %% Remove pilot
            
            y1 = remove_pilot(y_eq);
            
            %% Joint Decoding and Detection (SCMA + Polar + Deinterleaver)
            
            mhat_llr = JIDD(y1,polar_N,polar_K,FZlookup,K,V,M,N,CB,N0(iter_ebn0),h,iter_num,isInterleaver,interleaver,alpha);
            
            %**********************************************************
            llr = reshape(mhat_llr',1,V*polar_K);
            m_reshape = reshape(infobits', 1, polar_K*V);
            m_hat = llr<0;
            err = sum(m_hat~=m_reshape);
            Nerr(iter_ebn0) = Nerr(iter_ebn0) + err;
            Nbits(iter_ebn0) = Nbits(iter_ebn0) + length(m_reshape);
            %             Pb(iter_ebn0)=mean(abs((m_reshape)-Nbits(iter_ebn0)));
            fprintf('.')
        end
        %     figure(1);
        fprintf('\n')
        BER(iter_ebn0) = Nerr(iter_ebn0)/Nbits(iter_ebn0);
        
        
        
        
        fprintf('EbN0 is %d, have runned %d bits, found %d errors, BER=%.7f \n',EbN0(iter_ebn0),Nbits(iter_ebn0),Nerr(iter_ebn0),BER(iter_ebn0));
        
        
        
        
    end


% figure(1);
% SNR_db=0:0.25:7;
% SNR=10.^(SNR_db/10);
% P_theory=(0.5)*erfc(sqrt(2.*SNR)./sqrt(2));
% semilogy(SNR_db,Pb,'o',SNR_db,P_theory,'r-')
% title ('SNR Vs Probabolity of Error');
% xlabel ('SNR (dB)');
% ylabel ('Probability of Error');
% legend('practical curve','theoratical curve');
% grid on;
%% plot graph
toc
figure(1);
semilogy(EbN0,BER,'linewidth',1);
grid on;




% semilogy((iter_ebn0),packetErrorRate);
% hold on;
% grid on;
% xlabel('SNR (dB)');
% ylabel('PER');
% dataStr = arrayfun(@(x)sprintf('MCS %d',x),mcs,'UniformOutput',false);
% legend(dataStr);
% title('PER for DMG OFDM-PHY with AWGN channel');