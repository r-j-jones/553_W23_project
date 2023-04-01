% ====================================================================
% Sample code to replicate the experiments in
%
%   D. Pimentel-Alarcon, A. Biswas, C. Solis-Lemus
%   Adversarial Principal Component Analysis,
%   IEEE International Symposium on Information Theory (ISIT), 2017.
%
% This code shows one trial of the evolution of the angle between a
% subspace and its PCA estimate over time when corrupted with adversarial
% and random outliers.
%
% Written by: D. Pimentel-Alarcon.
% email: pimentelalar@wisc.edu
% Created: 2017
% =====================================================================

clear all; close all; clc;
rand('state',sum(100*clock));

d = 5;          % Ambient dimension
r = 4;          % Subspace dimension
n = 100;        % Number of initial datapoints
eps = 1e-3;     % Initial noise level
T = 500;       % Number of new vectors
lambda_r = 5;  % Signal to noise ratio (amount that each new vector will weight)
p = 0.05;       % Probability of a new vector being an outlier

% General form of the adversarial outlier. Keep four sign combinations
thetaStar = 1/2*acos(-1/lambda_r^2);
y_star = zeros(d,4);
cost = cos(thetaStar); sint = sin(thetaStar);
y_star(r,:) = [cost,cost,-cost,-cost];
y_star(r+1,:) = [sint,-sint,sint,-sint];

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

% Angles over time.
phi_rand = zeros(T,1);
phi_adv = zeros(T,1);

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
    phi_aux = acos(svd(Ux'*Ut_rand(:,1:r)));
    phi_rand(t) = phi_aux(r);
    Xt_rand = Ut_rand(:,1:r)*Lambdat_rand(1:r,1:r)/Lambdat_rand(r,r)*lambda_r;
    
    
    % ========== Update Subspace with adversarial outlier ==========
    % Check the four sign options and pick the "worst" to avoid tilting back
    % (see Remark 1)
    for i=1:4,
        [Ut_adv4{i},Lambdat_adv,~] = svd([Xt_adv,zt_adv(:,i)]);
        phi_aux = acos(svd(Ux'*Ut_adv4{i}(:,1:r)));
        phi_adv4(i) = phi_aux(r);
    end
    [phi_adv(t),i] = max(phi_adv4);
    Xt_adv = Ut_adv4{i}(:,1:r)*Lambdat_adv(1:r,1:r)/Lambdat_adv(r,r)*lambda_r;
    
end

% ===== Create and save figure =====
figure;
clrs = [0,.75]; %Colors.
hold on;
plot(phi_adv,'k-','LineWidth',4,'Color',repmat(clrs(1),1,3));
plot(phi_rand,'k-','LineWidth',4,'Color',repmat(clrs(2),1,3));
set(gca,'fontsize',15);
set(gca,'fontsize',15);
xlabel('Time','Interpreter','latex','fontsize',20);
ylabel('$\varphi$','Interpreter','latex','fontsize',25);
set(gca,'YTick',[0,pi/48,2*pi/48,3*pi/48,4*pi/48,5*pi/48,6*pi/48,7*pi/48,8*pi/48],'yticklabel',{'0','\pi/48','\pi/24','\pi/16','\pi/12','5\pi/48','\pi/8','7\pi/48','\pi/6'},'fontsize',15);
leg = legend('$(i)$ Adversarial','$(ii)$ Random');
set(leg,'FontSize',15,'Location','NorthWest','Interpreter','latex');
set(gcf,'PaperUnits','centimeters');
set(gcf, 'renderer','default');
set(gcf,'PaperSize',[12,10],'PaperPosition',[0,0,12,10]);
figurename = 'figure4left.pdf';
saveas(gcf,figurename);







