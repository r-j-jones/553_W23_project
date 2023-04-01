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