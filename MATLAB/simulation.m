Nt = 2;
K = 2;
weight = 10.^(-3:0.01:3);
weight = [weight;ones(size(weight))];

SNR_dB = 10;
tol = 1e-3;

H = sqrt(2) \ (randn(Nt,K) + 1j * randn(Nt,K));

n_sim = size(weight,2);
R = zeros(n_sim,K);
for n = 1:n_sim
    R1 = DPCrateRegion(weight(:,n),H,SNR_dB,tol);
    R(n,:) = R1.'; 
end

plot(R(:,1),R(:,2))
xlabel('R_1')
ylabel('R_2')