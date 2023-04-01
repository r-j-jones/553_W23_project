function [L,S,Uhat,Coeffs] = R2PCA_n(M, r, k, sigma)

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
%   k = noise dimension parameter (k>r)
%   sigma = noise level
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

if k<=r
    error('Must have k>r');
end

Uhat = LoR(M,r,k,sigma);        % 1st part of R2PCA: recover basis of low-rank component
Coeffs = Sp(M,Uhat,k,sigma,r);    % 2nd part of R2PCA: recover coefficients
L = Uhat*Coeffs;        % Recover low-rank compoment
S = M-L;                % Recover sparse component
end


% % % % Add in noisy case - 
% % % 




% ===== 1st part of R2PCA: recover basis of low-rank component =====
% ===== Try to find d-r uncorrupted (r+1)x(r+1) blocks (that are ===
% ===== independent). Each block will give us a projection of the ==
% ===== subspace. Then we "stitch" together all projections to =====
% ===== recover the subspace =======================================
function Uhat = LoR(M,r,k,noiselevel)

fprintf('\n---ON LoR function...---\n')

[d,N] = size(M);    % Dimensions of the problem
A = zeros(d,0);     % Matrix to store the information of the projections

for outerind=1:d-r

%     disp(outerind);

    % w_i == omega_i : r+1 random column indices
    wi = randsample(d,r+1);
    
    % k_i == kappa_i 
    %     - add new random samples from {1,2,...,d} one by
    %        one to kappa_i, only if they're not in omega_i

    valid_ki_range = setdiff(1:d, wi);
    ki = union(wi, randsample(valid_ki_range, k-r-1));



%     ki = wi;
%     while length(ki)<k
%         temp = randsample(d,1);
%         if ~ismember(temp,wi) && ~ismember(temp,ki)
%             ki = cat(1,ki,temp); 
%         end
%     %     disp(length(ki))
%     end

    if length(ki)~=length(unique(ki))
        disp('Error- length(ki)~=length(unique(ki))');
    end
    
    % extract kappa_i rows from data matrix M 
    Mki = M(ki,:);

    %%% Get _k_ random cols of M_kappa_i
    %%%  - get svdvals for the (r+1)th singular value
    %%%  - repeat with different random _k_ until (r+1)th svdval is <= noise level
    singval_rplus1 = noiselevel * 1e3;    % for while criteria
    cnt = 0;                            % 4 debugging
    trk = [];
    while singval_rplus1>noiselevel
        kprime = randsample(N,k);  % this is random "k" col inds in paper
        Mki_prime = Mki(:,kprime); % M'_kappa_i matrix
        stemp = svds(Mki_prime,r+1);  % the first r+1 svdvals of M'_kappa_i
        singval_rplus1 = stemp(end); % the r+1th svdval 
        cnt = cnt+1;
        trk(end+1)=singval_rplus1;
        if mod(cnt,2e3)==0
            fprintf(' -cnt=%d -min=%g\n',cnt,min(trk));
            disp((singval_rplus1-noiselevel)/noiselevel);
        end
%         if mod(cnt,20)==0, disp(cnt); end
    end
    % Get r-leading left singular vectors of M'_kappa_i (Alg1 lines 14-15)
    [Vtmp,Stmp,~] = svd(Mki_prime);
    Vki = Vtmp(:,1:r);
    
    % vi == v_i : extract subset of r entries of wi (Alg1 lines 16-17)
    kappa_inds = randsample( length(ki), r );
    vi = ki(kappa_inds);

    % vec of j values : find elements of ki not in vi (for-loop iterate in Alg1 line 18
    js = setdiff(ki,vi);
    
    % iterate through j entries (Alg1 line 18)
    for ind=1:length(js)
        % element of kappa_i NOT IN v_i
        j = js(ind);
        % wij == omega_ij = union(vi,j)  (Alg1 line 19)
        wij = cat(1,vi,j);
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
            disp('bad awij')
        end
    
%         if ind<length(js), clear awij wij Vwij awij; end
    end
end

% figure; imshow(A~=0); title('A~=0'); drawnow;

% Bottom right of page 5 in R2PCA paper:
%  We can thus use the matrix Uhat in R^dxr formed with the last r left 
%  singular vectors of A (which approximates ker AT ) to estimate of U
[Ua, Sa, ~] = svd(A);
Sa = diag(Sa);

Uhat = Ua(:,end-r+1:end);
% Uhat = null(A');

end



% ============ 2nd part of R2PCA: recover coefficients ============
% ====== Try to find r+1 uncorrupted entries in each column =======
% ====== This entries determine the coefficient of the column =====
function Coeffs = Sp(M,U,k,noiselevel,r)

disp('--On function Sp...-');

% r=5;


[d,N] = size(M);        % Dimensions of the problem

% r = size(U,2);          % rank of the low-rank component
Coeffs = zeros(r,N);    % Matrix to keep the coefficients

% ======== Look one column at a time ========
for j=1:N  
%     disp(j);
    respcnt = 0;
    valtrack = [];
    resp = 0;   % Have we already found r+1 uncorrupted entries in this column?
    if mod(j,5)==0, disp(['j=' num2str(j)]); end
    
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
        
    tic;
    bestcoeffs = zeros(r,1);
    minerr = 1e10;
    % ======== Start looking for uncorrupted entries ========
    while resp==0 && toc<1e+2/N
        
        % == Take r+1 random entries, and check if they are corrupted ==
        oi = randsample(d,k);
        Uoi = U(oi,:);
        xoi = M(oi,j);
        Coeffs(:,j) = (Uoi'*Uoi)\Uoi'*xoi;
        xoiPerp = xoi-Uoi*Coeffs(:,j);

        currerr = norm(xoiPerp)/norm(xoi);
        if currerr<minerr, bestcoeffs = Coeffs(:,j); end
        
        % == If the entries are uncorrupted, use them to obtain a
        % == coefficient and move on to the next column
        if norm(xoiPerp)/norm(xoi) < noiselevel
            resp = 1; disp('resp=1! yaya!')
        end
        
    end
    if resp==0
        Coeffs(:,j) = bestcoeffs;
    end
end



end





