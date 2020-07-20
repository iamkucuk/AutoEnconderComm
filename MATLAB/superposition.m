n_users = 2;
n_messages = 1e6;

SNRdb_vec = -10:30;

b = 2;
M = 2^(b * n_users);
bits = randi(2,[n_users, n_messages * b]) -1;
symbols = zeros(n_users * b, n_messages);
for user = 1:n_users
    symbols((user - 1) * b + 1 : user * b, :) = reshape(bits(user, :), b, []);
end

symbols = 2.^(n_users * b-1:-1:0) * symbols;
x = qammod(symbols, M,'gray','UnitAveragePower',true);
ber = zeros(n_users, length(SNRdb_vec));

for ii = 1:length(SNRdb_vec)
    SNRdb = SNRdb_vec(ii);
    SNR_dB_sym = SNRdb+10 * log10(b);
    
    for user = 1:n_users
        y = awgn(x,SNR_dB_sym,'measured');
        z = qamdemod(y,M,'gray','UnitAveragePower',true);

        xhat = de2bi(z.').';
        xhat = xhat(end:-1:1,:);
        xhat = xhat((user - 1) * b + 1 : user * b, :);
        xhat = xhat(:);
        ber(user, ii) = mean(xhat.'~=bits(user, :));
    end
        
end

semilogy(SNRdb_vec, ber.');