clear all;close all;clc
n_users = 2;
n_messages = 1e6;

SNRdb_vec = -10:30;

coeff = (1/2).^(0:n_users-1);

b = 2;
M = 2^b;
bits = randi(2,[n_users, n_messages * b]) -1;
symbols = zeros(n_users, n_messages);
x = 0;
for n = 1:n_users
    symbols(n, :) = 2.^(b-1:-1:0) * reshape(bits(n,:),b,[]);
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
   
%     for user = 1:n_users
    y = awgn(x,SNR_dB_sym,'measured');
    z = qamdemod(y,M^2,'UnitAveragePower',true); 
    for jj = [2, 3, 10, 11]
        z(z == jj) = (z(z == jj) + 2) * 10;
    end
    for jj = [4, 5, 12, 13]
        z(z == jj) = (z(z == jj) - 2) * 10;
    end
    z(z > M^2) = z(z > M^2)/10;
    ser(1, ii) = mean(floor(z / M)~=symbols(1, :));
    ser(2, ii) = mean(mod(z, M)~=symbols(2, :));
%     end
        
end
figure
semilogy(SNRdb_vec, ber.');

hebe = 1 - ber;
hube = sum(hebe);
figure;
semilogy(SNRdb_vec, hebe.');
