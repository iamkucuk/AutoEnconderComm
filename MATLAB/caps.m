clc;
close all;
clear all;
mT = 2;
mR = 2;
ITER = 1000;
SNRdB = [0:25];
SNR = 10.^(SNRdB/10);
C_SISO = zeros(1,length(SNR));
C_SIMO = zeros(1,length(SNR));
C_MISO = zeros(1,length(SNR));
C_MIMO = zeros(1,length(SNR));
for ite = 1:ITER
    h_SISO = (randn +j*randn)/sqrt(2);
    h_SIMO = (randn(mR,1)+j*randn(mR,1))/sqrt(2);
    h_MISO = (randn(1,mT)+j*randn(1,mT))/sqrt(2);
    h_MIMO = (randn(mR,mT)+j*randn(mR,mT))/sqrt(2);
    for K = 1:length(SNR)
        C_SISO(K) = C_SISO(K) + log2(1+ SNR(K)*norm(h_SISO)^2);
        C_SIMO(K) = C_SIMO(K) + log2(1+ SNR(K)*norm(h_SIMO)^2);
        C_MISO(K) = C_MISO(K) + log2(1+ SNR(K)*norm(h_MISO)^2/mT);
        C_MIMO(K) = C_MIMO(K) + log2(abs(det(eye(mR)+SNR(K)*h_MIMO*h_MIMO'/mT)));
    end
end
C_SISO = C_SISO/ITER;
C_SIMO = C_SIMO/ITER;
C_MISO = C_MISO/ITER;
C_MIMO = C_MIMO/ITER;
plot(SNRdB,C_SISO,'r - .',SNRdB,C_SIMO,'b - o',SNRdB,C_MISO,'m',SNRdB,C_MIMO,'k - *')
legend('SISO','SIMO','MISO','MIMO')
xlabel('SNR in dB')
ylabel('Capacity (b/s/Hz)')
title('Capacity Vs. SNR')
grid;