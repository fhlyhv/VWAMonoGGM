function [V,nu,run_time,Eomega,Elambda,Adj,idl] = VWA_Mono_GGM(XDat)

% variational Wishart approximation using natural gradients
% Yu Hang, Oct, 2018, NTU

tolm = 1e-3;
tolr = 1e-4;
maxIter = 1e6;
tic;

%% initialization
[n,p] = size(XDat);
pe = (p-1)*p/2;

idl = zeros(pe,1);
idu = zeros(pe,1);
% idr = zeros(pe,1);
% idc = zeros(pe,1);
k = 1;
for j = 1:p
    for i = j+1:p
        idl(k) = (j-1)*p+i;
        idu(k) = (i-1)*p+j;
%         idr(k) = i;
%         idc(k) = j;
        k = k+1;
    end
end

S = cov(XDat,1);
nS = n*S;

nu = n+p+1;
if n > p
    Vinv = S*nu;
else
    Vinv = (0.99*S+1e-2*speye(p))*nu;
end
V = inv(Vinv);

a = pe/2;
b = a/50;

d = 0.5*ones(pe,1);
Elambda = zeros(pe,1);
ELAMBDA = zeros(p);
EK0 = 0;
VARKdnu = V.^2+diag(V)*diag(V)';
ll_old = 0;

eta0 = 5; %1e-5;% % 0.9;
%% 
for kappa = 1:maxIter
    
    
    Eomega = a/b;
    idd = find(d>=10)';
    if ~isempty(idd)
        for i = idd
            Elambda(i) = Lentz_Algorithm(d(i));
        end
    end
    Elambda(d<10) = exp(d(d<10)).*expint(d(d<10));
    Elambda = 1./(Elambda.*d)-1;   %0.5./d; %
    ELAMBDA(idl) = Eomega*Elambda;
    ELAMBDA(idu) = ELAMBDA(idl);
    
    
    
    tmp = 1/2/(MvPolyGamma(nu/2,p,1)/2-p/nu)*sum(sum(ELAMBDA.*VARKdnu));
    natgradVinv = nS+((nu+1)*ELAMBDA.*V+spdiags(ELAMBDA*diag(V),0,p,p))+(tmp/nu-1)*Vinv;
    natgradnu =  n+tmp+p+1-nu;
    natgradVinv = (natgradVinv+natgradVinv')/2;
    
    gradnu = (n-nu+p+1)/4*MvPolyGamma(nu/2,p,1)+p/2-sum(sum(V.*nS))/2 ...
        -sum(sum(ELAMBDA.*((2*nu+1)*V.^2+diag(V)*diag(V)')))/4;
    gradV = (n+p+1)/2*Vinv - nu/2*nS - (nu^2+nu)/2*V.*ELAMBDA - nu/2*diag(ELAMBDA*diag(V));
    gradVinv = -V*gradV*V;
    
    sumgrad = gradnu*natgradnu + sum(sum(gradVinv.*natgradVinv));
    if sumgrad < 0
        sumgrad;
    end
    
    eta = eta0;
    ELBO0 = ELBO_Wishart_mono(nu,Vinv,p,n,nS,ELAMBDA);
    while 1
        nutmp = nu + eta*natgradnu;
        Vinvtmp = Vinv + eta*natgradVinv;
        ELBOtmp = ELBO_Wishart_mono(nutmp,Vinvtmp,p,n,nS,ELAMBDA);
        if ELBOtmp >= ELBO0 + 0.01*eta*sumgrad
            Vinv = Vinv + eta*natgradVinv;
            Vinv = (Vinv+Vinv')/2;
            V = inv(Vinv);
            V = (V+V')/2;
            
            nu = nu + eta*natgradnu;
            break;
        end
        eta = eta/2;
    end
    
    VARKdnu = V.^2+diag(V)*diag(V)';
    EK = nu*V;
    EK2 = EK.^2+nu*VARKdnu;
    d = Eomega/2*EK2(idl);
    
    b = sum(Elambda.*EK2(idl))/2;
    
    
    if rem(kappa,10) == 0
        diffmax = max(abs(EK(:)-EK0(:)));
        diffr = norm(EK(:)-EK0(:))/norm(EK0(:));
        ll = Elambda./(Elambda+1);
        diffll = max(abs(ll - ll_old));
        fprintf('#no. of iterations =%d, difmax = %d, difrm = %d, diflbd = %d\n',kappa,diffmax,diffr,diffll);
        if diffr < tolr && diffmax < tolm && diffll < tolr
            break;
        else
            EK0 = EK;
            ll_old = ll;
        end
    end
end

t = toc;
run_time = t;
fprintf("VWA-MonoGGM is done, elapsed time is %d seconds\n", t);

fprintf("estimate adjacency matrix by thresholding lambda / (1 + lambda)...\n");
tic;
ll = Elambda ./ (1+Elambda);
[~, fx, x, ~] = kde(ll, 4096, 0, 1);
idx = find(x > 1e-2 & x < 0.6);
fx = fx(idx);
x = x(idx);
fx_min = min(fx);
q = find(fx <= fx_min);
figure;
plot(x, fx);
hold on; plot(x(q(1)), 0, 'r+');
legend('kernel density', 'selected threshold');
title('Density function of <\lambda_{jk}> / (<\lambda_{jk}> + 1)');

thr = x(q(1)) / (1 - x(q(1)));
Adj = ELAMBDA / Eomega < thr;
t = toc;
fprintf("adjacency marix has been estimated, elapsed time is %d seconds\n", t);
run_time = run_time + t;
end

function E1b =  Lentz_Algorithm(x)
    epsilon1 = 1e-30;
    epsilon2 = 1e-7;
    f_prev = epsilon1;
    C_prev = epsilon1;
    D_prev = 0;
    delta = 2+epsilon2;
    j = 1;
    while (delta-1>=epsilon2 || 1-delta >= epsilon2)
        j = j+1;
        tmp1 = x+2*j-1;
        tmp2 = (j-1)^2;
        D_curr = 1/(tmp1-tmp2*D_prev);
        C_curr = tmp1-tmp2/C_prev;
        delta = C_curr*D_curr;
        f_curr = f_prev*delta;
        f_prev = f_curr;
        C_prev = C_curr;
        D_prev = D_curr;
    end
    E1b =  1/(x+1+f_curr);
end
    
    