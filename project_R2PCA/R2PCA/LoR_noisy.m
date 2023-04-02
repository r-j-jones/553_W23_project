% ===== 1st part of R2PCA: recover basis of low-rank component =====
% ===== Try to find d-r uncorrupted (r+1)x(r+1) blocks (that are ===
% ===== independent). Each block will give us a projection of the ==
% ===== subspace. Then we "stitch" together all projections to =====
% ===== recover the subspace =======================================
function Uhat = LoR_noisy(M,r,k,noiselevel, verbose)

% set verbose==1 to display messages
% set verbose>1 to make plots (for "educational" purposes only)

if nargin<5, verbose=0; end

atic = tic;

fprintf('\n---ON LoR function...---\n')

[d,N] = size(M);    % Dimensions of the problem
A = zeros(d,0);     % Matrix to store the information of the projections

if verbose==1, fprintf(' --on outerind (_) of %d : \n',d-r); end

if verbose>1
    fig = figure('position',[2474 -50 841 306]);
end

for outerind=1:d-r
    if verbose==1
        otic = tic;
        fprintf('%d',outerind);
    end

    % w_i == omega_i : r+1 random column indices
    wi = randsample(d,r+1);
    wi = sort(wi);
    
    % k_i == kappa_i 
    %-- add new random samples from {1,2,...,d}, only if they're not in omega_i
    valid_ki_range = setdiff(1:d, wi);
    inds_for_ki = randsample(valid_ki_range, k-r-1);
    ki = union(wi, inds_for_ki);
    ki = sort(ki);

    if length(ki)~=length(unique(ki)) || length(ki)~=k
        warning('Error- kappa_i has incompatible dimension (should be length k)');
    end
    
    % extract kappa_i rows from data matrix M 
    Mki = M(ki,:);

    if verbose>1
        [fig, ax1, ax2, jetmap, dat] = create_sval_plot(fig,[d,N], ki);
        llog = [];
        pause(1);
    end

    %%% Get _k_ random cols of M_kappa_i
    %%%  - get svdvals for the (r+1)th singular value
    %%%  - repeat with different random _k_ until (r+1)th svdval is <= noise level
    singval_rplus1 = noiselevel * 1e3;    % for while criteria
    cnt = 0; trk = [];                    % 4 debugging
    while singval_rplus1>noiselevel
        kprime = randsample(N,k);  % this is random "k" col inds in paper
        kprime = sort(kprime);
        Mki_prime = Mki(:,kprime); % M'_kappa_i matrix
        svals = svds(Mki_prime, r+1);  % the first r+1 svdvals of M'_kappa_i
        singval_rplus1 = svals(end); % the r+1th svdval 
        cnt = cnt+1;   trk(end+1)=singval_rplus1;

%         if mod(cnt,2e3)==0 && verbose==1
%             fprintf(' -cnt=%d -min=%g\n',cnt,min(trk));
%             disp((singval_rplus1-noiselevel)/noiselevel);
%         end

        if verbose>1
            [llog] = update_sval_plot(ax1, ax2, llog, dat, kprime, svals, noiselevel, jetmap);
            pause(0.01);
        end
    end

    if verbose>1
        update_sval_plot(ax1, ax2, llog);
        prompt = "Enter anything to proceed ";
        useless = input(prompt);
    end

    % Get r-leading left singular vectors of M'_kappa_i (Alg1 lines 14-15)
    [Vtmp,Stmp,~] = svd(Mki_prime);
    Vki = Vtmp(:,1:r);

    
    
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



function [fig, ax1, ax2, cmap, dat] = create_sval_plot(fig, dims, ki)
    fig.Visible='on';
    subplot(121); ax1= gca;
    subplot(122); ax2= gca;
    cmap = jet(10);
    
    dat = zeros(dims);
    dat(ki,:)=10;
    axes(ax1);
    imshow(dat,'border','tight','Parent',ax1);
    colormap(cmap);
    title('\kappa_i rows')
end


function [lll] = update_sval_plot(ax1, ax2, lll, dat, kprime, svals, noiselevel, cmap)

% if 3 args, then change line opacity/alpha
if nargin==3
%     lll=findall(ax2,'type','line');
%     lll=findobj(ax2,'Type','line','-not',{'Tag','linear','-or','Tag','xaxis'});

    for ii=1:length(lll)-1
        set(lll(ii),'Color',[get(lll(ii),'Color') 0.2]);
    end

else

    % if nargin<8
    %     llog = [];
    % end
    
    axes(ax1); cla;
    dat(:,kprime)=8;
    imshow(dat,'border','tight','Parent',ax1);
    titlestr = ['k'' cols'];
    title(titlestr); colormap(cmap);
    
    axes(ax2); hold on;
    lll(end+1) = plot(svals,'-o','Color',[0 0 0.8],'LineWidth',2);
    titlestr = ['\sigma_{r+1} M_{k}''=' sprintf('%g',svals(end))];
    title(titlestr); 
    xx=xlim; yy=ylim;
    ylim([-0.1 yy(2)]);
    plot(xx,[noiselevel noiselevel],'--r','LineWidth',2,'Tag','noiselevel');
    plot(xx,[0 0],'-k','Tag','xaxis');
    legend('\sigma',['noise lvl=' sprintf('%g',noiselevel)],'Location','best');
    drawnow;
    
    %to clean old lines
    if length(lll)>25
        delete(lll(1));
        lll=lll(2:end);
    end

end

end

