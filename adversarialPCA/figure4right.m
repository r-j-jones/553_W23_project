% ====================================================================
% Sample code to replicate the experiments in
%
%   D. Pimentel-Alarcon, A. Biswas, C. Solis-Lemus
%   Adversarial Principal Component Analysis,
%   IEEE International Symposium on Information Theory (ISIT), 2017.
%
% This code shows the angle between a subspace and its {CA estimate after T
% updates as a function of lambda_r, which essentially determines the
% influence of each new datum, and and the fraction of outliers p.
%
% Written by: D. Pimentel-Alarcon.
% email: pimentelalar@wisc.edu
% Created: 2017
% =====================================================================

clear all; close all; clc;
rand('state',sum(100*clock));

d = 5;                  % Ambient dimension
r = 4;                  % Subspace dimension
n = 100;                % Number of initial datapoints
eps = 1e-3;             % Initial noise level
T = 500;                % Number of new vectors
Lambda_r = 1.5:0.5:10;    % Signal to noise ratio (amount that each new vector will weight)
P = 0.025:0.025:0.6;    % Probability of a new vector being an outlier
numTrials = 100;

% Auxiliary variables to save results
PHI_rand = zeros(length(Lambda_r),length(P),numTrials);
PHI_adv = zeros(length(Lambda_r),length(P),numTrials);

for trial=1:numTrials,
    for lambda=1:length(Lambda_r),
        lambda_r = Lambda_r(lambda);
        
        % General form of the adversarial outlier. Keep four sign combinations
        thetaStar = 1/2*acos(-1/lambda_r^2);
        y_star = zeros(d,4);
        cost = cos(thetaStar); sint = sin(thetaStar);
        y_star(r,:) = [cost,cost,-cost,-cost];
        y_star(r+1,:) = [sint,-sint,sint,-sint];
        
        for pp=1:length(P),
            p = P(pp);
            
            % Initial data
            X = randn(d,r);         % Basis of the true subspace
            Ux = orth(X);           % Orthonormal basis of the true subspace
            B = randn(r,n);         % Coefficients of initial vectors.
            E = eps * randn(d,n);   % Noise
            Xprime = X*B + E;       % Initial data matrix
            
            % First step, t=0
            [Ut,Lambdat,~] = svd(Xprime);
            % Normalize to always have smallest singular value = lambda_r
            Xt_rand = Ut(:,1:r)*Lambdat(1:r,1:r)/Lambdat(r,r)*lambda_r;
            Xt_adv = Xt_rand;
            
            % Auxiliary structures to check all sign options
            Ut_adv4 = cell(4,1);
            phi_adv4 = zeros(4,1);
            
            for t=1:T,
                
                % ========== Create new datapoint ==========
                
                if rand>p, % Create inlier (same inlier for both cases)
                    zt_rand = X * randn(r,1) + eps*randn(d,1);
                    zt_rand = zt_rand/norm(zt_rand);
                    zt_adv = repmat(zt_rand,1,4);
                else % Create Outliers
                    zt_rand = randn(d,1);
                    zt_rand = zt_rand/norm(zt_rand);
                    
                    % Ubar is orthonormal basis of ambient space whose first columns
                    % are Ut. Ubar is needed to change coordinate system of the
                    % adversarial outlier.
                    [Ubar,~,~] = svd(Xt_adv);
                    zt_adv = Ubar * y_star;
                end
                
                % ========== Update Subspace with random outlier ==========
                % Compute Ut, Lambdat phi and Zt, normalized to always have smallest
                % singular value = lambda_r.
                [Ut_rand,Lambdat_rand,~] = svd([Xt_rand,zt_rand]);
                Xt_rand = Ut_rand(:,1:r)*Lambdat_rand(1:r,1:r)/Lambdat_rand(r,r)*lambda_r;
                
                
                % ========== Update Subspace with adversarial outlier ==========
                % Check the four sign options and pick the "worst" to avoid tilting back
                % (see Remark 1)
                for i=1:4,
                    [Ut_adv4{i},Lambdat_adv,~] = svd([Xt_adv,zt_adv(:,i)]);
                    phi_aux = acos(svd(Ux'*Ut_adv4{i}(:,1:r)));
                    phi_adv4(i) = phi_aux(r);
                end
                [phi_adv,i] = max(phi_adv4);
                Xt_adv = Ut_adv4{i}(:,1:r)*Lambdat_adv(1:r,1:r)/Lambdat_adv(r,r)*lambda_r;
                
            end
            
            % Keep track of results
            phi_aux = acos(svd(Ux'*Ut_rand(:,1:r)));
            PHI_rand(lambda,pp,trial) = phi_aux(r);
            PHI_adv(lambda,pp,trial) = phi_adv;
            
            fprintf('trial=%d, lambda_r=%1.1d, p=%1.1d, phi_rand=%1.1d, phi_adv=%1.1d \n',...
                trial,lambda_r,p,phi_aux(r),phi_adv);
            
        end
    end
end

% ===== Create and save transition diagram =====
figure;
imagesc(mean(PHI_adv,3));
colormap(gray);
set(gca,'YDir','normal');
set(gca,'XTick',4:4:length(P),'xticklabel',P(4:4:end),'fontsize',15);
set(gca,'YTick',2:4:length(Lambda_r),'yticklabel',Lambda_r(2:4:end),'fontsize',15);
xlabel('$p$','Interpreter','latex','fontsize',20);
ylabel('$\lambda_r$','Interpreter','latex','fontsize',25);
set(gcf,'PaperUnits','centimeters');
set(gcf, 'renderer','default');
set(gcf,'PaperSize',[12,10],'PaperPosition',[0,0,12,10]);
figurename = 'figure4right.pdf';
saveas(gcf,figurename);







