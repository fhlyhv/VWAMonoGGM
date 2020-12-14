function ELBO = ELBO_Wishart_mono(nu,Vinv,P,n,nS,ELAMBDA)
[~,q] = chol(Vinv);

if q ~= 0
    ELBO = -Inf;
else
    V = inv(Vinv);
    % V = (V+V')/2;
    logdetV = logdet(V);
    psipnuo2 = MvPolyGamma(nu/2,P,0);
    % nElogdetK = n/2*(psipnuo2+logdetV);
    HpnElogdetK = (P+1+n)/2*logdetV+logMvGamma(nu/2,P)-(nu-P-n-1)/2*psipnuo2+nu*P/2;
    EK = nu*V;
    VK = nu*(V.*V+diag(V)*diag(V)');
    
    ELBO = -sum(sum(nS.*EK))/2-sum(sum(ELAMBDA.*(EK.^2+VK)))/4+HpnElogdetK;

end





