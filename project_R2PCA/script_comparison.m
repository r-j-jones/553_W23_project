
%% configure paths

% compare different RPCA methods: 
%  (1) - R2PCA
%  (2) - RPCA-ALM
%
% Use R2PCA script/code to generate data 
% %    (noiseless definitely, noisy? --look at RPCA-ALM paper for noise
%       handling...)


clear all; close all; warning ('off','all'); % clc;
rand('state',sum(100*clock));

if ~isdeployed
    addpath('/Users/robertjones/Desktop/W23/553/project/R2PCA/noiseless/utils');
    addpath(genpath('/Users/robertjones/Desktop/W23/553/project/RPCA+MC_codes/inexact_alm_rpca'));
    addpath('/Users/robertjones/Desktop/W23/553/project');
end

% ===========================  ============================
%% Set parameters

% Input:
%   
%   d,N = size of the matrix is d x N
%   r = rank of low-rank component.
%   p = Probability of nonzero entry in sparse component
%   v = variance of the sparse entries
%   mu = coherence parameter of low-rank component
%   T = number of trials
%   k = size of expanded block/window for noisy-case
%   NoiseLvl = noise level of AGWN

d = 100;            % Size of the matrix = d x N
N = d;
r = 5;              % rank of low-rank component
p = 0.05;           % probability of sparse outliers
mu = 8;             % Coherence: value in [1,d/r]
v = 10;             % variance of the sparse entries %v=10 used in paper
T = 1;              % Number of trials
k = 10;             % Size of block in noisy-case (k>r)
sigma_n = 1e-3;     % Noise level (standard deviation (sigma) of Gauss dist)

UseNoisyData = true;

verbose = 1;        % display messages, if debugging


% fprintf('d=%d\tN=%d\tr=%g\tp=%g\nmu=%g\tv=%g\tk=%g\tsigman=%g\n', d,N,r,p,mu,v,k,sigma_n);
fprintf('\n---- ---- ---- --------------- ---- ---- ---- \n')
fprintf('---- ---- ---- RPCA PARAMETERS ---- ---- ---- \n')
fprintf('-Matrix size (d x N):        \t d = %d, N = %d\n',d,N);
fprintf('-Low-rank dim:               \t r = %d \n',r);
fprintf('-Prob of sp outliers:        \t p = %g \n',p);
fprintf('-Var. or sparse outliers:    \t v = %g \n',v);
fprintf('-Coherence:                  \t %s = %g \n',char(956),mu);
fprintf('-R2PCA noisy search wind sz: \t k = %d \n',k);
fprintf('-Gauss. noise std dev:       \t %s_noise = %g \n',char(963),sigma_n);
fprintf('-# of trials:                \t T = %d \n',T);
fprintf('-Verbose:                    \t verbose = %d \n',verbose);
fprintf('---- ---- ---- --------------- ---- ---- ---- \n\n')


% ===========================  ============================
%% Generate synthetic/simulated data
fprintf('\n======== GENERATING SIMULATED DATA ==========\n');


s = ceil(p*N);               %number of corrupted entries per row
fprintf(' -# of corrupted entries per row = %g \n',s);

if s > (N-r)/(2*(r+1))
    warning('The # of corrupted entries in S is large (S not sparse enough)');
end

% ===================== GENERATE U MATRIX ============================
% ======= Low-rank basis within range of coherence parameter mu =======
fprintf(' -Generating low-rank subspace U with coherence ~%s=%g \n',char(956),mu);
U = basisWcoherence(d,r,mu-.5);
while coherence(U)>mu+.5
    U = basisWcoherence(d,r,mu-.5);
end
U = orth(U);
mu = coherence(U);
fprintf('\t -Final coherence value: %s = %g \n',char(956),mu);

% ===================== GENERATE L MATRIX (& THETA) ============================
fprintf(' -Generating low-rank matrix L\n');
fprintf('\t L = U*%s \n',char(920));
fprintf('\t %s = random coeffs of low-rank component\n',char(920));
Theta = randn(r,N);     % Coefficients of low-rank component
L = U*Theta;            % Low-rank component

% ===================== GENERATE S MATRIX ============================
% ========== Sparse matrix with s corrupted entries per row ==========
fprintf(' -Generating sparse outlier matrix S\n');
fprintf(' \t with %g corrupted entries per row \n',s);
S = zeros(d,N);
for i = 1:d
    S(i,randsample(N,s)) = v*randn(s,1);
end
% Verify that each column has at least r+1 uncorrupted entries
for j=1:N
    idx = find(S(:,j));
    nonZeros = length(idx);
    if nonZeros>d-r-1
        newZeros = nonZeros-(d-r-1);
        S(idx(randsample(nonZeros,newZeros)),j) = 0;
    end
end

% ===================== GENERATE W (NOISE) MATRIX ============================
% =============== Construct synthetic additive noise matrix ==========
fprintf(' -Generating Gauss noise matrix W\n');
fprintf(' \t with %s_noise = %g \n',char(963),sigma_n);

% W = sigma_n^2 * randn(d,N);    % as done when generating S above
W = normrnd(0,sigma_n^2,[d,N]);
% W1 = sigma_n^2 * randn(d,N); 
% W2 = normrnd(0,sigma_n,[d,N]);

% ===================== GENERATE M (DATA/OBS) MATRIX ============================
% ======================= Mixed matrix =======================
fprintf(' -Creating observation data matrix M \n');
fprintf(' \t via M = L + S + W \n');
if UseNoisyData
    fprintf('--USING NOISY DATA--\n');
    M = L + S + W;
else
    fprintf('--USING NOISE-FREE DATA--\n');
    M = L + S;
end


%% Run R2PCA recon

% ================== Run R2PCA ==================
fprintf('Running R2PCA...');
tic1 = tic;


% [Lhat,Shat,Uhat,Thetahat] = R2PCA_n(M, r, k, NoiseLvl);

% make sure k>r, error if not
if k<=r, error('Must have k>r'); end

Uhat = LoR_noisy(M, r, k, sigma_n, verbose);        % 1st part of R2PCA: recover basis of low-rank component
[Coeffs,minerrors] = Sp_noisy(M, Uhat, k, sigma_n, r, verbose);    % 2nd part of R2PCA: recover coefficients
Lhat = Uhat * Coeffs;        % Recover low-rank compoment
Shat = M - Lhat;   

elaptime = toc(tic1);
Lreconerr = norm(L-Lhat,'fro')/norm(L,'fro');
Sreconerr = norm(S-Shat,'fro')/norm(S,'fro');
Ureconerr = norm(U-Uhat,'fro')/norm(U,'fro');
Creconerr = norm(Theta-Coeffs,'fro')/norm(Theta,'fro');

if 0 == 1
    fprintf('==== FINISHED RECON ====\n');
    fprintf('\tTotal elapsed time:\t%g seconds\n',elaptime);
    fprintf('\tL reconst. error:  \t%g \n',Lreconerr);
    fprintf('\tS reconst. error:  \t%g \n',Sreconerr);
    fprintf('\tU reconst. error:  \t%g \n',Ureconerr);
    fprintf('\tTheta rec. error:  \t%g \n',Creconerr);
    fprintf('==================================================\n');
    fprintf('d=%d\tN=%d\tr=%g\tp=%g\nmu=%g\tv=%g\tk=%g\tsigman=%g\n', d,N,r,p,mu,v,k,sigma_n);
    fprintf('==================================================\n');
end


%% Run RPCA-ALM recon

% [A_hat E_hat iter] = inexact_alm_rpca(D, lambda, tol, maxIter, rho);
% D = A + E ;  --> A === L low rank, E === S sparse outliers corruption

lambda = 1/sqrt(d) ;
lambda = 1e-3;
tol = 1e-9;
maxIter = -1;
rho = mu;
rho = 0.1;

btic = tic ;
[L_dual, S_dual, numIter] = inexact_alm_rpca_rj(M, lambda, tol, maxIter, rho) ;
elaptimeALM = toc(btic) ;
    
LreconerrALM = norm(L-L_dual,'fro')/norm(L,'fro');
SreconerrALM = norm(S-S_dual,'fro')/norm(S,'fro');
% UreconerrALM = norm(U-Uhat,'fro')/norm(U,'fro');
% CreconerrALM = norm(Theta-Coeffs,'fro')/norm(Theta,'fro');

if 0 == 1
    fprintf('==== FINISHED ALM RECON ====\n');
    fprintf('\tALM Total elapsed time:\t%g seconds\n',elaptimeALM);
    fprintf('\tALM L reconst. error:  \t%g \n',LreconerrALM);
    fprintf('\t   ALM L rank:  \t%g \n',rank(L_dual));
    fprintf('\tALM S reconst. error:  \t%g \n',SreconerrALM);
    % fprintf('\tALM U reconst. error:  \t%g \n',UreconerrALM);
    % fprintf('\tALM Theta rec. error:  \t%g \n',CreconerrALM);
    fprintf('==================================================\n');
    fprintf('d=%d\tN=%d\tr=%g\tp=%g\nmu=%g\tv=%g\tk=%g\tsigman=%g\n', d,N,r,p,mu,v,k,sigma_n);
    fprintf('==================================================\n');

    % disp('Relative error in estimate of A') ;
    % error = norm(L_dual-L,'fro')/norm(L,'fro');
    % disp(error);
    % disp('Relative error in estimate of E') ;
    % disp(norm(S_dual-S,'fro')/norm(S,'fro')) ;
    % disp('Number of iterations') ;
    % disp(numIter) ;
    % disp('Rank of estimated A') ;
    % disp(rank(L_dual)) ;
    % disp('0-norm of estimated E') ;
    % disp(length(find(abs(S_dual)>0))) ;
    % disp('|D-A-E|_F');
    % disp(norm(M_noisefree-L_dual-S_dual,'fro'));
    % disp('Time taken (in seconds)') ;
    % disp(tElapsed) ;
    % disp('obj value');
    % disp(sum(svd(L_dual))+lambda*sum(sum(abs(S_dual))));
    % disp('original value');
    % disp(sum(svd(L))+lambda*sum(sum(abs(S))));
end



%% Display results

res = [Lreconerr, LreconerrALM; rank(Lhat), rank(L_dual); Sreconerr, SreconerrALM; elaptime, elaptimeALM];
resTable = array2table(res,'VariableNames',{'R2PCA','RPCA-ALM'},'RowNames',{'Lhat error','Lhat rank','Shat error','Time'});
disp(resTable)

