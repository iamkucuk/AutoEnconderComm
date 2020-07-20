n_users = 2;
n_messages = 1e6;

SNRdb_vec = -10:30;

t = .25.^(0:n_users - 1);
h = [1, 1];
h = sort(h, "ascending");

b = 2;
M = 2^(b * n_users);
bits = randi(2,[n_users, n_messages * b]) -1;
x = 0;
for user = 1:n_users
    symbols = reshape(bits(user, :), b, []);
    symbols = 2.^(b-1:-1:0) * symbols;
    x_n = qammod(symbols, 2^b,'gray','UnitAveragePower',true);
    x = x + t(user) * x_n;
end

ber = zeros(n_users, length(SNRdb_vec));

for ii = 1:length(SNRdb_vec)
    SNRdb = SNRdb_vec(ii);
    SNR_dB_sym = SNRdb+10 * log10(b);
    
    for user = 1:n_users
        y = awgn(h(user) * x,SNR_dB_sym,'measured');
        for jj = 1:user
            
            z = qamdemod(y,2^b,'gray','UnitAveragePower',true);
            
            z_mod = qammod(z, 2^b,'gray','UnitAveragePower',true);
            y = y - z_mod;
            
        end
        
        xhat = de2bi(z.').';
        xhat = xhat(end:-1:1,:);
        xhat = xhat(:);
        ber(user, ii) = mean(xhat.'~=bits(user, :));
        
    end
        
end

semilogy(SNRdb_vec, ber.');