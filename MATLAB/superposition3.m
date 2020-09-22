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

symbol_places = zeros(M);
for ii = 1:M
    x2 = qammod(ii - 1, M, 'UnitAveragePower', true);
    for jj = 1:M
        symbol_places(ii, jj) = x2 + coeff(n) *  qammod(jj - 1, M,'UnitAveragePower',true);
    end
end

figure
scatter(real(symbol_places(:)),imag(symbol_places(:)));
ber = zeros(n_users, length(SNRdb_vec));
ser = zeros(n_users, length(SNRdb_vec));

symbols_repeated = repmat(symbol_places(:).', [1, n_messages]);

for ii = 1:length(SNRdb_vec)
    SNRdb = SNRdb_vec(ii);
    SNR_dB_sym = SNRdb;
   
    for user = 1:n_users
        y = awgn(x,SNR_dB_sym,'measured');
        
        y_repeated = repmat(y, [M^2, 1]);
        y_repeated = y_repeated(:).';
        
        dists = abs(y_repeated - symbols_repeated);
        
        dists = reshape(dists, 4, 4, []);
        
        [min_dists, sym1] = min(dists); 
        [~, sym2] = min(min_dists);
        
        new_sym1 = zeros(1, n_messages);
        for jj = 1:n_messages
            new_sym1(jj) = sym1(1, sym2(jj), jj);
        end
        
        sym1 = new_sym1 - 1;
        sym2 = sym2(:).' - 1;
        
        ser(1, ii) = mean(sym1 ~= symbols(1, :));
        ser(2, ii) = mean(sym2 ~= symbols(2, :));

    end
        
end
figure
semilogy(SNRdb_vec, ser.');

hebe = 1 - ser;
hube = sum(hebe);
figure;
semilogy(SNRdb_vec, hube.');
