%%
%For 2 users

P1 = 
sigma = 1;

r_1 = log(1 + abs(h_1)^2 * P_1 / sigma^2);
r_2 = log(1 + abs(h_2)^2 * P_2 / (abs(h_2)^2 * P_1 + sigma^2));