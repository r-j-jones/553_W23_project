function [err,time,mu] = runExperiment_Noise(d,N,r,p,v,mu,NoiseSigma,k)

% ====================================================================
% This code generates a low-rank plus sparse matrix within a specified
% coherencerange and sparsity level (see Figure 2), and then decomposes
% this matrix using the R2PCA algorithm introduced in
%
%   D. Pimentel-Alarcon, R. Nowak
%   Random Consensus Robust PCA,
%   International Conference on Artificial Intelligence and Statistics
%   (AISTATS), 2017.
%
% Input:
%   
%   d,N = size of the matrix is d x N
%   r = rank of low-rank component.
%   p = Probability of nonzero entry in sparse component
%   v = variance of the sparse entries
%   mu = coherence parameter of low-rank component
%
% Output:
%
%   err = error of low-rank component
%   time = computation time 
%   mu = exact coherence of low-rank component
%
% Written by: D. Pimentel-Alarcon.
% email: pimentelalar@wisc.edu
% Created: 2017
% =====================================================================

% mu is in [1,d/r].
s = ceil(p*N);               %number of corrupted entries per row

% ======= Low-rank basis within range of coherence parameter mu =======
U = basisWcoherence(d,r,mu-.5);
while coherence(U)>mu+.5
    U = basisWcoherence(d,r,mu-.5);
end
U = orth(U);
mu = coherence(U);
fprintf('\n Final mu = %1.1d \n \t',mu);

Theta = randn(r,d);     % Coefficients of low-rank component
L = U*Theta;            % Low-rank component

% ========== Sparse matrix with s corrupted entries per row ==========
S = zeros(d,N);
for i = 1:d,
    S(i,randsample(N,s)) = v*randn(s,1);
end

% Verify that each column has at least r+1 uncorrupted entries
for j=1:N,
    idx = find(S(:,j));
    nonZeros = length(idx);
    if nonZeros>d-r-1
        newZeros = nonZeros-(d-r-1);
        S(idx(randsample(nonZeros,newZeros)),j) = 0;
    end
end

% =============== Construct synthetic additive noise matrix ==========
W = NoiseSigma^2 * randn(d,N);
% W = normrnd(0,Sigma,[d,N]);

% ======================= Mixed matrix =======================
M = L + S + W;


% ================== Run R2PCA ==================
fprintf('Running R2PCA...');
tic1 = tic;
[Lhat,Shat,Uhat,Thetahat] = R2PCA_n(M, r, k, NoiseSigma);
% [Lhat,~] = R2PCA(M,r);
time = toc(tic1);
err = norm(L-Lhat,'fro')/norm(L,'fro');

end


% ===== Auxiliary function to obtain a basis of a subspace within a =====
% ===== range of a specified coherence. This is done by increasing  =====
% ===== the magnitude of somerows of the basis U.                   =====
function U = basisWcoherence(d,r,mu)
U = randn(d,r);
c = coherence(U);
i = 1;
it = 1;
while c<mu
    U(mod(i,d+1),:) = U(i,:)/(10^it);
    i = i+1;
    c = coherence(U);
    if i==d+1,
        i = 1;
        it = it+1;
    end
end
end


% ===== Auxiliary function to compute coherence paraemter =====
function mu = coherence(U)
P = U/(U'*U)*U';
[d,r] = size(U);

Projections = zeros(d,1);
for i=1:d,
    ei = zeros(d,1);
    ei(i) = 1;
    Projections(i) = norm(P*ei,2)^2;
end

mu = d/r * max(Projections);
end



% % ===== Auxiliary function to create noise matrix =====
% function W = noise_synth(d,N,Sigma)
% W = Sigma^2 * randn(d,N);
% % W = normrnd(0,Sigma,[d,N]);
% end




















