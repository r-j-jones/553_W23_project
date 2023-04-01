%%%%% A script for testing?debugging noisy-R2PCA implementation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% !!!!!! README !!!!!! 
%  - This is for testing purposes.
%  - Manually set parameters in %% Set parameters block below and run

function do_all_noisy_testing


% =========================== GENERAL SETUP ============================
% ======================================================================
% In their paper, for simulated data generation, it seems they used :
%     d = N = 100       % dimension of data matrix (dxN)
%     r = 5             % rank of L, low rank matrix
%  
clear all; close all; warning ('off','all'); clc;
rand('state',sum(100*clock));


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
k = 8;             % Size of block in noisy-case (k>r)
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


% ================== Run R2PCA ==================
fprintf('Running R2PCA...');
tic1 = tic;


% [Lhat,Shat,Uhat,Thetahat] = R2PCA_n(M, r, k, NoiseLvl);

% make sure k>r, error if not
if k<=r, error('Must have k>r'); end

Uhat = LoR(M, r, k, sigma_n, verbose);        % 1st part of R2PCA: recover basis of low-rank component
[Coeffs,minerrors] = Sp(M, Uhat, k, sigma_n, r, verbose);    % 2nd part of R2PCA: recover coefficients
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
    if i==d+1
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
for i=1:d
    ei = zeros(d,1);
    ei(i) = 1;
    Projections(i) = norm(P*ei,2)^2;
end

mu = d/r * max(Projections);
end


% ===== 1st part of R2PCA: recover basis of low-rank component =====
% ===== Try to find d-r uncorrupted (r+1)x(r+1) blocks (that are ===
% ===== independent). Each block will give us a projection of the ==
% ===== subspace. Then we "stitch" together all projections to =====
% ===== recover the subspace =======================================
function Uhat = LoR(M,r,k,noiselevel, verbose)

if nargin<5, verbose=0; end

atic = tic;

fprintf('\n---ON LoR function...---\n')

[d,N] = size(M);    % Dimensions of the problem
A = zeros(d,0);     % Matrix to store the information of the projections

if verbose==1, fprintf(' --on outerind (_) of %d : \n',d-r); end

for outerind=1:d-r
    if verbose==1
        otic = tic;
        fprintf('%d',outerind);
    end

    % w_i == omega_i : r+1 random column indices
    wi = randsample(d,r+1);
    wi = sort(wi);
    
    % k_i == kappa_i 
    %     - add new random samples from {1,2,...,d} one by
    %        one to kappa_i, only if they're not in omega_i

    valid_ki_range = setdiff(1:d, wi);
    inds_for_ki = randsample(valid_ki_range, k-r-1);
    ki = union(wi, inds_for_ki);
    ki = sort(ki);

%     ki = wi;
%     while length(ki)<k
%         temp = randsample(d,1);
%         if ~ismember(temp,wi) && ~ismember(temp,ki)
%             ki = cat(1,ki,temp); 
%         end
%     %     disp(length(ki))
%     end

    if length(ki)~=length(unique(ki)) || length(ki)~=k
        warning('Error- kappa_i has incompatible dimension (should be length k)');
    end
    
    % extract kappa_i rows from data matrix M 
    Mki = M(ki,:);

    %%% Get _k_ random cols of M_kappa_i
    %%%  - get svdvals for the (r+1)th singular value
    %%%  - repeat with different random _k_ until (r+1)th svdval is <= noise level
    singval_rplus1 = noiselevel * 1e3;    % for while criteria
    cnt = 0; trk = [];                    % 4 debugging
    while singval_rplus1>noiselevel
        kprime = randsample(N,k);  % this is random "k" col inds in paper
        kprime = sort(kprime);
        Mki_prime = Mki(:,kprime); % M'_kappa_i matrix
        stemp = svds(Mki_prime, r+1);  % the first r+1 svdvals of M'_kappa_i
        singval_rplus1 = stemp(end); % the r+1th svdval 
        cnt = cnt+1;   trk(end+1)=singval_rplus1;
        if mod(cnt,2e3)==0 && verbose==1
            fprintf(' -cnt=%d -min=%g\n',cnt,min(trk));
            disp((singval_rplus1-noiselevel)/noiselevel);
        end
    end
    % Get r-leading left singular vectors of M'_kappa_i (Alg1 lines 14-15)
    [Vtmp,Stmp,~] = svd(Mki_prime);
    Vki = Vtmp(:,1:r);
    
    % % % % % 3/28/23 - RJ - realized I was using kappa_i instead of
    %                   omega_i here... but didnt change results in Sp()..
    % vi == v_i : extract subset of r entries of wi (Alg1 lines 16-17)
    wi_inds = randsample( length(wi), r );
    vi = wi(wi_inds);
    vi = sort(vi);

    % vec of j values : find elements of ki not in vi (for-loop iterate in Alg1 line 18
    js = setdiff(ki,vi);
    
    % iterate through j entries (Alg1 line 18)
    for ind=1:length(js)
        % element of kappa_i NOT IN v_i
        j = js(ind);
        % wij == omega_ij = union(vi,j)  (Alg1 line 19)
        wij = union(vi,j);
        wij = sort(wij);
        % row indices to extract from V_kappa_i
        v_ind = ismember(ki,wij);
        % V_omega_ij matrix
        Vwij = Vki(v_ind,:);
        % nonzero vec in ker(transpose(V_omega_ij)) (Alg1 lines 20-21
        awij = null(Vwij');
        %double check awij is nonzero vector
        if ~isempty(awij) && (nnz(awij==0)~=numel(awij))
            % insert awij into A (Alg1 lines 22-23)
            aij = zeros(d,1);
            aij(wij) = awij;
            A(:,end+1) = aij;
        else
            if verbose==1, disp('bad awij'); end
        end
    
%         if ind<length(js), clear awij wij Vwij awij; end
    end
    if verbose==1
        otoc = toc(otic);
        fprintf(' %d s, ',round(otoc));
    end
end

% figure; imshow(A~=0); title('A~=0'); drawnow;

% Bottom right of page 5 in R2PCA paper:
%  We can thus use the matrix Uhat in R^dxr formed with the last r left 
%  singular vectors of A (which approximates ker AT ) to estimate of U
[Ua, Sa, ~] = svd(A);
Sa = diag(Sa);

Uhat = Ua(:,end-r+1:end);
% Uhat = orth(Uhat);
% Uhat = null(A');

atoc = toc(atic);
fprintf('---LoR complete---\n');
fprintf('  - elap time: %d sec\n',round(atoc));

end



% ============ 2nd part of R2PCA: recover coefficients ============
% ====== Try to find r+1 uncorrupted entries in each column =======
% ====== This entries determine the coefficient of the column =====
function [Coeffs,minerrors] = Sp(M,U,k,noiselevel,r, verbose)
    if nargin<6, verbose=0; end

    disp('--On function Sp...-');
    
    [d,N] = size(M);        % Dimensions of the problem
    
    % r = size(U,2);          % rank of the low-rank component
    Coeffs = zeros(r,N);    % Matrix to keep the coefficients

    respcnt = 0; minerrors = zeros(N,1);
    
    % ======== Look one column at a time ========
    for j=1:N  
        if mod(j,5)==0 && verbose==1, fprintf('j=%d\n',j); end               
%         errtrack = [];
        
        minerr = 1e10;  
        bestcoeffs = zeros(r,1);
        resp = 0;   % Have we already found k uncorrupted entries in this column?
        tic; % ======== Start looking for uncorrupted entries ========
        while resp==0 && toc<1e+2/N
            
            % == Take k random entries, and check if they are corrupted ==
            oi = randsample(d,k);
            Uoi = U(oi,:);
            xoi = M(oi,j);
            Coeffs(:,j) = (Uoi'*Uoi)\Uoi'*xoi;
            xoiPerp = xoi-Uoi*Coeffs(:,j);
    
            currerr = norm(xoiPerp)/norm(xoi);
%             errtrack(end+1)=currerr;
            if currerr<minerr
                minerr = currerr;
                bestcoeffs = Coeffs(:,j); 
            end
            
            % == If the entries are uncorrupted, use them to obtain a
            % == coefficient and move on to the next column
            if norm(xoiPerp)/norm(xoi) < noiselevel
                resp = 1; 
%                 if verbose==1, disp('resp=1! yaya!'); end
            end   
        end
        if resp==0 
            if verbose==1, fprintf(' using best coeffs for j=%d\n',j); end
            Coeffs(:,j) = bestcoeffs; 
            respcnt = respcnt+1;
            minerrors(j) = minerr;
        end
    end

    if verbose==1, fprintf(' successful cols found = %d/%d\n',N-respcnt,N); end
end



%     % ======== Start looking for uncorrupted entries ========
    %     while resp==0
    %         respcnt = respcnt+1;
    % 
    %         kappa = randsample(d, k);
    % %         m_kappa = M_j(kappa);
    % 
    %         U_kappa = U(kappa,:);
    %         m_kappa = M(kappa,j);
    %         % Projection 
    %         Coeffs(:,j) = (U_kappa'*U_kappa)\U_kappa'*m_kappa;
    % %         Coeffs(:,j) = inv(U_kappa'*U_kappa)*U_kappa'*m_kappa;
    %         M_kappa_perp = m_kappa-U_kappa*Coeffs(:,j);
    % 
    %         % == If the entries are uncorrupted, use them to obtain a
    %         % == coefficient and move on to the next column
    %         val = norm(M_kappa_perp)/norm(m_kappa) ;
    %         valtrack = cat(1,valtrack,val);
    %         if val < noiselevel
    %             resp = 1;
    %         end
    %     end


