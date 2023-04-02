
function script_noisy


% =========================== GENERAL SETUP ============================
% ======================================================================
% In their paper, for simulated data generation, it seems they used :
%     d = N = 100       % dimension of data matrix (dxN)
%     r = 5             % rank of L, low rank matrix
%  
clear all; close all; warning ('off','all'); clc;
rand('state',sum(100*clock));

if ~isdeployed
    addpath('/Users/robertjones/Desktop/W23/553/project/R2PCA/noiseless/utils');
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
mu = 5;             % Coherence: value in [1,d/r]
v = 10;             % variance of the sparse entries %v=10 used in paper
T = 1;              % Number of trials
k = 10;             % Size of block in noisy-case (k>r)
sigma_n = 1e-3;     % Noise level (standard deviation (sigma) of Gauss dist)

verbose = 1;        % display messages, if debugging

fprintf('d=%d\tN=%d\tr=%g\tp=%g\nmu=%g\tv=%g\tk=%g\tsigman=%g\n', d,N,r,p,mu,v,k,sigma_n);

% [err,Time,mu] = runExperiment_Noise(d, N, r, p, v, mu, NoiseLvl,k);


% ===========================  ============================
%% Run R2PCA reconstruction

s = ceil(p*N);               %number of corrupted entries per row
if s > (N-r)/(2*(r+1))
    warning('# of corrupted entries in S is large (S not sparse enough)');
end

% ===================== GENERATE U MATRIX ============================
% ======= Low-rank basis within range of coherence parameter mu =======
U = basisWcoherence(d,r,mu-.5);
while coherence(U)>mu+.5
    U = basisWcoherence(d,r,mu-.5);
end
U = orth(U);
mu = coherence(U);
fprintf('\n Final coherence value: mu = %1.1d \n \t',mu);

% ===================== GENERATE L MATRIX (& THETA) ============================
Theta = randn(r,N);     % Coefficients of low-rank component
L = U*Theta;            % Low-rank component

% ===================== GENERATE S MATRIX ============================
% ========== Sparse matrix with s corrupted entries per row ==========
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
% W = sigma_n^2 * randn(d,N);    % as done when generating S above
W = normrnd(0,sigma_n^2,[d,N]);
% W1 = sigma_n^2 * randn(d,N); 
% W2 = normrnd(0,sigma_n,[d,N]);

% ===================== GENERATE M (DATA/OBS) MATRIX ============================
% ======================= Mixed matrix =======================
M = L + S + W;

params.sigman = sigma_n;
params.r = r;
params.p = p;
% [fig] = DisplayInputData(M, L, S, W, params);


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