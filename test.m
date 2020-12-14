load('artidata500_4p_2p.mat')

[V,nu,t,Eomega,Elambda,Adj,idl] = VWA_Mono_GGM(XDat);

precision = sum(Adj(idl) > 0 & Ktrue(idl) ~= 0) / sum(Adj(idl) > 0);
recall = sum(Adj(idl) > 0 & Ktrue(idl) ~= 0) / sum(Ktrue(idl) ~= 0);

fprintf('precision = %d, recall = %d\n', full(precision), full(recall));

%% When some data are missing, please call the function VWA_Mono_GGM_missing
fprintf('add missing data\n');
[n, p] = size(XDat);
id_missing = randperm(n * p, round(n * p * 0.05));
XDat(id_missing) = NaN;
[V,nu,t,Eomega,Elambda,Adj,idl] = VWA_Mono_GGM_missing(XDat);

precision = sum(Adj(idl) > 0 & Ktrue(idl) ~= 0) / sum(Adj(idl) > 0);
recall = sum(Adj(idl) > 0 & Ktrue(idl) ~= 0) / sum(Ktrue(idl) ~= 0);
fprintf('precision = %d, recall = %d\n', full(precision), full(recall));