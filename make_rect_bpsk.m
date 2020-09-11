% Script for creating a simple rectangular-pulse BPSK signal.
% Chad M. Spooner
% March 2015

% Set up figure windows.

for ind=1:2
    hf = figure(ind);
    clf;
    set(hf, 'units', 'normalized');
    set(hf, 'paperpositionmode', 'auto');
    set(hf, 'position', [0.2 0.05 0.6 0.35]);
end
figure(2);
set(gcf, 'position', [0.2 0.5 0.6 0.35]);

%% Initialize key variables.

T_bit = 10;              % 1/T_bit is the bit rate
num_bits = 4000;         % Desired number of bits in generated signal
fc = 0.05;               % Desired carrier frequency(normalized units)
N0_dB = -10.0;           % Noise spectral density(average noise power)
Power_dB = 0.0;          % Signal power in decibels
N_psd = 128;             % Number of frequencies to use in PSD estimate

%% Create the baseband signal.

% Create bit sequence.

bit_seq = randi([0 1], [1 num_bits]);

% Create symbol sequence from bit sequence.

sym_seq = 2*bit_seq - 1;
zero_mat = zeros((T_bit - 1), num_bits);
sym_seq = [sym_seq ; zero_mat];
sym_seq = reshape(sym_seq, 1, T_bit*num_bits);

% Create pulse function.

p_of_t = ones(1, T_bit);

% Convolve bit sequence with pulse function.

s_of_t = filter(p_of_t, [1], sym_seq);

% Plot time-domain waveform.

figure(1);
hp = plot(s_of_t(1:200));
set(gca, 'ylim', [-1.2 1.2]);
set(hp, 'linewidth', 2);
grid on;
xlabel('Sample Index');
ylabel('Signal Amplitude');
title('Time-Domain Plot of Rectangular-Pulse BPSK (T_{bit} = 10, f_c = 0)');
print -djpeg99 'rect_bpsk_time_domain.jpg'

%% Frequency-shift the baseband signal and add noise.

% Apply the carrier frequency.

e_vec = exp(sqrt(-1)*2*pi*fc*[1:length(s_of_t)]);
x_of_t = s_of_t .* e_vec;

% Add noise.

n_of_t = randn(size(x_of_t)) + sqrt(-1)*randn(size(x_of_t));
noise_power = var(n_of_t);
N0_linear = 10^(N0_dB/10);
pow_factor = sqrt(N0_linear / noise_power);
n_of_t = n_of_t * pow_factor;
y_of_t = x_of_t + n_of_t;

%% Estimate PSD and plot.

num_blocks = floor(length(y_of_t) / N_psd);
S = y_of_t(1, 1:(N_psd*num_blocks));
S = reshape(S,  N_psd, num_blocks);
I = fft(S);
I = I .* conj(I);
I = sum(I.');
I = fftshift(I);
I = I /(num_blocks * N_psd);

freq_vec = [0:(N_psd-1)]/N_psd - 0.5;

figure(2);
hp = plot(freq_vec, 10*log10(I));
set(hp, 'linewidth', 2);
grid on;
xlabel('Frequency (Normalized)');
ylabel('PSD(dB)');
title('Estimated Power Spectrum for Rectangular-Pulse BPSK (T_{bit} = 10, f_c = 0.05)');
print -djpeg99 'rect_bpsk_psd.jpg'

%% Check power of PSD against known sum of noise and signal powers.

meas_pow = sum(I) *(1.0/N_psd);
Power_linear = 10^(Power_dB/10.0);
fprintf('PSD-measured power is %.5e, known total power is %.5e\n', ...
    meas_pow, N0_linear + Power_linear);



