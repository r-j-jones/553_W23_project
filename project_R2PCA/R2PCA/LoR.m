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
        disp('');
        
    end
    
end

% Stitch all the projections into the whole subspace, given by
% ker(A).
Uhat = null(A');

end
