clear all;close all;clc
n_users = 2;
n_messages = 1e6;

SNRdb_vec = -10:30;

coeff = (1/2).^(0:n_users-1);

b = 2;
M = 2^b;
bits = randi(2,[n_users, n_messages * b]) -1;
x = 0;
for n = 1:n_users
    symbols = 2.^(b-1:-1:0) * reshape(bits(n,:),b,[]);
    x1 = qammod(symbols, M,'gray','UnitAveragePower',true);
    x = x + coeff(n) * x1;
end
figure
scatter(real(x),imag(x));
ber = zeros(n_users, length(SNRdb_vec));

for ii = 1:length(SNRdb_vec)
    SNRdb = SNRdb_vec(ii);
    SNR_dB_sym = SNRdb;
   
    for user = 1:n_users
        y = awgn(x,SNR_dB_sym,'measured');
        for n = 1:user
            z = qamdemod(y,M,'gray','UnitAveragePower',true);
            x1 = qammod(z, M,'gray','UnitAveragePower',true);
            y = y - coeff(n) * x1;
        end
        xhat = de2bi(z.').';
        xhat = xhat(end:-1:1,:);
%         xhat = xhat((user - 1) * b + 1 : user * b, :);
        xhat = xhat(:);
        ber(user, ii) = mean(xhat.'~=bits(user, :));
    end
        
end
figure
semilogy(SNRdb_vec, ber.');

hebe = 1 - ber;
hube = sum(hebe);
figure;
semilogy(SNRdb_vec, hebe.');
