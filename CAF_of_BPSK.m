% Edit: 2020/07/06
% Reference: https://cyclostationary.blog/2015/09/28/the-cyclic-autocorrelation-for-rectangular-pulse-bpsk/
clear all; close all; clc;
ifig = 1;

%% Initialize key variables
Fs = 30e3;
Fc = 5e3;
Rsym = 1e3;
sps = Fs / Rsym;
nsym = 4e3;
M = 2; 
iniPhase = 0;
snr = 10;
psType = 0; % 0 for rectangular pulse, 1 for srrc pulse
sigType = 1; % 0 for complex received signal, 1 for real received signal

%% Create baseband signal
msg_seq = randi([0,M-1],nsym,1);
sym_seq = pskmod(msg_seq,M,iniPhase);
sym_us_seq = upsample(sym_seq,sps);
if psType == 0
    ps = ones(sps,1); % shaping pulse: rectangular pulse
else
    ps =  rcosdesign(0.3,4,sps);% shaping pulse: srrc pulse
end
tx_baseband = filter(ps,1,sym_us_seq);
tseq = [0:length(tx_baseband)-1].' / Fs;
len_sig = length(tseq);

%% Frequency-shift the baseband signal and add noise
carrier = exp(1j * 2 * pi * Fc * tseq);
tx_passband_cmp = tx_baseband .* carrier; % complex signal
tx_passband_real = real(tx_passband_cmp); % real signal


if sigType == 0
    rxsig = awgn(tx_passband_cmp,snr,'measured');
else
    rxsig = awgn(tx_passband_real,snr,'measured');
end
    
figure(ifig); ifig = ifig + 1;
idx_plot = 1:sps*20;
subplot(2,1,1); plot(real(tx_baseband(idx_plot)),'linewidth',2); 
xlabel('Sample Index'); ylabel('Signal Amplitude'); title('Time-Domain Plot of Rectangular-Pulse Baseband BPSK Signal')
subplot(2,1,2); plot(tx_passband_real(idx_plot),'linewidth',1);
xlabel('Sample Index'); ylabel('Signal Amplitude'); title('Time-Domain Plot of Rectangular-Pulse Passband BPSK Signal')


%% Calculate CAF and Plot
%%% 1st version
% dtau = 1e-4;
% maxTau = 20 * dtau;
% maxTau_ind = maxTau * Fs;
% tau_seq = -maxTau : dtau : maxTau;
% tau_ind = round(tau_seq * Fs);
% nTau = length(tau_seq);
% t_ind_max = 2^(floor(log2(length(rxsig)-2*maxTau_ind)));
% t_ind = [1 : t_ind_max];
% nFFT = length(t_ind);
% fre_seq = (-nFFT / 2 : nFFT/2-1) * Fs / nFFT;
% CAF_nonconj = zeros(nFFT,nTau);
% CAF_conj = zeros(nFFT,nTau);
% 
% for aiter = 1 : length(tau_ind)
%     tau = tau_ind(aiter);
%     sig0 = rxsig(t_ind + maxTau_ind);
%     sig1 = rxsig(t_ind + maxTau_ind + tau);
%     AF_nonconj = sig0 .* conj(sig1);
%     AF_conj = sig0 .* sig1;
%     caf_nonconj = fftshift(fft(AF_nonconj,nFFT)) / nFFT;
%     caf_conj = fftshift(fft(AF_conj,nFFT)) / nFFT;
%     CAF_nonconj(:,aiter) = caf_nonconj;
%     CAF_conj(:,aiter) = caf_conj;
% end

%%% modified version based on Dr. Spooner's comment
dtau = 1/Fs;
maxTau = length(ps) / Fs;
maxTau_ind = maxTau * Fs;
tau_seq = -maxTau : dtau : maxTau;
tau_ind = round(tau_seq * Fs);
nTau = length(tau_seq);
t_ind_max = 2^(floor(log2(length(rxsig)-2*maxTau_ind)));
t_ind = [1 : t_ind_max];
nFFT = length(t_ind);
fre_seq = (-nFFT / 2 : nFFT/2-1) * Fs / nFFT;
CAF_nonconj = zeros(nFFT,nTau);
CAF_conj = zeros(nFFT,nTau);

for aiter = 1 : length(tau_ind)
    tau = tau_ind(aiter);
    sig0 = rxsig(t_ind + maxTau_ind);
    sig1 = rxsig(t_ind + maxTau_ind + tau);
    AF_nonconj = sig0 .* conj(sig1);
    AF_conj = sig0 .* sig1;
    caf_nonconj = fftshift(fft(AF_nonconj,nFFT)) / nFFT;
    caf_conj = fftshift(fft(AF_conj,nFFT)) / nFFT;
    CAF_nonconj(:,aiter) = caf_nonconj;
    CAF_conj(:,aiter) = caf_conj;
end


%%% Plot
figure(ifig); ifig = ifig + 1; 
set(gcf,'position',[560,480,1200,450]);
[Tau,Fre] = meshgrid(tau_seq,fre_seq);
subplot(2,2,1); surf(Fre/1e3,Tau*1e3,abs(CAF_nonconj),'EdgeColor','none'); xlim([-12,12]);
xlabel('$\alpha$(kHz)','Interpreter','latex'); ylabel('$\tau(ms)$','Interpreter','latex'); zlabel('Magnitude'); title('Cyclic Autocorrelation');
set(gca,'View',[170,40]);
subplot(2,2,2); surf(Fre/1e3,Tau*1e3,abs(CAF_nonconj)); xlim([-12,12]); 
xlabel('$\alpha$(kHz)','Interpreter','latex'); ylabel('$\tau(ms)$','Interpreter','latex'); zlabel('Magnitude'); title('Cyclic Autocorrelation');
set(gca,'View',[0,0]);
subplot(2,2,3); surf(Fre/1e3,Tau*1e3,abs(CAF_conj),'EdgeColor','none'); xlim([-12,12]); set(gca,'XDir','reverse');
xlabel('$\alpha$(kHz)','Interpreter','latex'); ylabel('$\tau(ms)$','Interpreter','latex'); zlabel('Magnitude'); title('Conjugate Cyclic Autocorrelation');
set(gca,'View',[-12,46]);
subplot(2,2,4); surf(Fre/1e3,Tau*1e3,abs(CAF_conj)); xlim([-12,12]); set(gca,'XDir','reverse');
xlabel('$\alpha$(kHz)','Interpreter','latex'); ylabel('$\tau(ms)$','Interpreter','latex'); zlabel('Magnitude'); title('Conjugate Cyclic Autocorrelation');
set(gca,'View',[0,0],'XDir','reverse');
if sigType == 0
    tstr = sprintf('SNR=%d dB, Complex Received Signal',snr);
else
    tstr = sprintf('SNR=%d dB, Real Received Signal',snr);
end
suptitle(tstr);