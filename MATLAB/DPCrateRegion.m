function [R] = DPCrateRegion(weight,H,P_max_dB,tolarence)
% H are the MAC channels of the system
[Nt,K] = size(H);
P = 10^(P_max_dB/10);
weight = weight(:);
[weight,index] = sort(weight,'descend');
H = H(:,index);

Q = rand(K,1);
Q = P * Q /sum(Q);

mu = [weight;0];
mu = mu(1:end-1) - mu(2:end);
eps = 1;
f1 = fun(1,Q,P,H,mu,1); 
while eps > tolarence
    grad = zeros(K,1);
    for jj = 1:K
        for ii = jj:K
            coeff = eye(Nt);
            for ll = 1:ii
                coeff = coeff + H(:,ll) * Q(ll) * H(:,ll)';
            end
            grad(jj) = grad(jj) + mu(ii) * real((H(:,jj)' / (coeff) * H(:,jj)));
        end
    end
    [~,jj] = max(grad);
    opt = optimoptions('fmincon','Display','off');
    t = fmincon(@(x) fun(x,Q,P,H,mu,jj),0.5,[],[],[],[],0,1,[],opt );
    Q = t * Q;
    Q(jj) = Q(jj) + (1-t) * P;
    f2 = fun(1,Q,P,H,mu,1);
    eps = abs(f2-f1);
    f1 = f2;
end
R = zeros(K,1);
coeff = eye(Nt);
for ii = 1:K
    coeff = coeff + H(:,ii) * Q(ii) * H(:,ii)';
    R(ii) = abs(log(det(coeff)));
end
R = [0;R];
R = R(2:end) - R(1:end-1);
R(index) = R;
end

function f = fun(t,Q,P,H,mu,jj)
    [Nt,K] = size(H);
    coeff = eye(Nt);
    Q = t * Q;
    Q(jj) = Q(jj) + (1-t) * P;
    f = 0;
    for ii = 1:K
        coeff = coeff + H(:,ii) * Q(ii) * H(:,ii)';
        f = f - mu(ii) * abs(log(det(coeff)));
    end
end