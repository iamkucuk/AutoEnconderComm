clear all;close all;clc
n_users = 2;
n_messages = 1e6;

SNRdb_vec = -10:30;

coeff = (1/2).^(0:n_users-1);

b = 2;
M = 2^b;
symbols = randi(M, [n_users, n_messages]) - 1;
x = 0;
for n = 1:n_users
    x1 = qammod(symbols(n, :), M,'UnitAveragePower',true);
    x = x + coeff(n) * x1;
end
figure
scatter(real(x),imag(x));
ber = zeros(n_users, length(SNRdb_vec));
ser = zeros(n_users, length(SNRdb_vec));
for ii = 1:length(SNRdb_vec)
    SNRdb = SNRdb_vec(ii);
    SNR_dB_sym = SNRdb;
   
    for user = 1:n_users
        y = awgn(x,SNR_dB_sym,'measured');
        for n = 1:user
            z = qamdemod(y,M,'UnitAveragePower',true);
            x1 = qammod(z, M,'UnitAveragePower',true);
            y = y - coeff(n) * x1;
        end
        ser(user, ii) = mean(symbols(user, :) ~= z);
    end
        
end
figure
semilogy(SNRdb_vec, ser.');

hebe = 1 - ser;
hube = sum(hebe);
figure;
semilogy(SNRdb_vec, hube.');
