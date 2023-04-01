function [L,S] = R2PCA(M,r)

% ====================================================================
% Decomposes a matrix M into its low-rank and sparse components using
% the simplest version of the R2PCA algorithm introduced in
%
%   D. Pimentel-Alarcon, R. Nowak
%   Random Consensus Robust PCA,
%   International Conference on Artificial Intelligence and Statistics
%   (AISTATS), 2017.
%
% Input:
%   
%   M = matrix with a combination of low-rank plus sparse component
%   r = dimension of low-rank component.
%
% Output:
%
%   L = Low-rank component of M
%   S = Sparse component of M
%
% Written by: D. Pimentel-Alarcon.
% email: pimentelalar@wisc.edu
% Created: 2017
% =====================================================================

Uhat = LoR(M,r);        % 1st part of R2PCA: recover basis of low-rank component
Coeffs = Sp(M,Uhat);    % 2nd part of R2PCA: recover coefficients
L = Uhat*Coeffs;        % Recover low-rank compoment
S = M-L;                % Recover sparse component
end


% ===== 1st part of R2PCA: recover basis of low-rank component =====
% ===== Try to find d-r uncorrupted (r+1)x(r+1) blocks (that are ===
% ===== independent). Each block will give us a projection of the ==
% ===== subspace. Then we "stitch" together all projections to =====
% ===== recover the subspace =======================================
function Uhat = LoR(M,r)

[d,N] = size(M);    % Dimensions of the problem
A = [];             % Matrix to store the information of the projections

tic;
% ======== Start looking for uncorrupted blocks ========
while rank(A)<d-r && toc<=1e+2,
    % = Select a random (r+1)x(r+1) block and check if it is corrupted =
    oi = randsample(d,r+1);
    oj = randsample(N,r+1);
    
    % == If the block is uncorrupted, keep it to obtain a projection ==
    if rank(M(oi,oj))==r,
        aoi = null(M(oi,oj)');
        A(oi,end+1) = aoi;
    else
%         disp('');
        
    end
    
end

% Stitch all the projections into the whole subspace, given by
% ker(A).
Uhat = null(A');

end



% ============ 2nd part of R2PCA: recover coefficients ============
% ====== Try to find r+1 uncorrupted entries in each column =======
% ====== This entries determine the coefficient of the column =====
function Coeffs = Sp(M,U)

[d,N] = size(M);        % Dimensions of the problem
r = size(U,2);          % rank of the low-rank component
Coeffs = zeros(r,N);    % Matrix to keep the coefficients

% ======== Look one column at a time ========
for j=1:N,
    
    resp = 0;   % Have we already found r+1 uncorrupted entries in this column?
    
    tic;
    % ======== Start looking for uncorrupted entries ========
    while resp==0 && toc<1e+2/N,
        
        % == Take r+1 random entries, and check if they are corrupted ==
        oi = randsample(d,r+1);
        Uoi = U(oi,:);
        xoi = M(oi,j);
        Coeffs(:,j) = (Uoi'*Uoi)\Uoi'*xoi;
        xoiPerp = xoi-Uoi*Coeffs(:,j);
        
        % == If the entries are uncorrupted, use them to obtain a
        % == coefficient and move on to the next column
        if norm(xoiPerp)/norm(xoi) < 1e-9,
            resp = 1;
        end
        
    end
end

end





