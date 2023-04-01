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