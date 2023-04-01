% ===== 1st part of R2PCA: recover basis of low-rank component =====
% ===== Try to find d-r uncorrupted (r+1)x(r+1) blocks (that are ===
% ===== independent). Each block will give us a projection of the ==
% ===== subspace. Then we "stitch" together all projections to =====
% ===== recover the subspace =======================================
function Uhat = LoR_noisy(M,r,k,noiselevel, verbose)

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