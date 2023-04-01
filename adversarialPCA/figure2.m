% ====================================================================
% Sample code to replicate the experiments in
%
%   D. Pimentel-Alarcon, A. Biswas, C. Solis-Lemus
%   Adversarial Principal Component Analysis,
%   IEEE International Symposium on Information Theory (ISIT), 2017.
%
% This code shows the angle phi between a 1-dimensional subspace X and its
% PCA estimate Z when corrupted with a unit-norm outlier, as a function of
% the angle theta that determines the location of the outlier and the
% energy in the subspace, given by lambda (see Figure 1 for intuition).
%
% Written by: D. Pimentel-Alarcon.
% email: pimentelalar@wisc.edu
% Created: 2017
% =====================================================================
clear all; close all; clc;

lambda = [1.1,1.25,1.5,2,5];        % energy in the subspace
N = 500;                            % number of points to plot
theta = linspace(0,pi/2,N);         % angle of the outlier
phi = zeros(length(lambda),N);      % angle between true subspace X and PCA estimate Z
thetaStar = 1/2*acos(-1./lambda.^2);% angle that maximizes phi
phiStar = zeros(length(lambda),1);

% Compute phi for all values of theta
for l=1:length(lambda)
    for i=1:N
        Z = [lambda(l) cos(theta(i)); 0 sin(theta(i))];
        [v,~,~] = svd(Z);
        v = v(:,1);
        phi(l,i) = acos(v(1)/norm(v));
    end
    
    % Compute phi for thetaStar
    Z = [lambda(l) cos(thetaStar(l)); 0 sin(thetaStar(l))];
    [v,~,~] = svd(Z);
    v = v(:,1);
    phiStar(l) = acos(v(1)/norm(v));
    
end


% ===== Create and save plot =====
figure;
clrs = [0,.4,.5,.6,.7]; %To present the colors in the right order
hold on;
for l=1:length(lambda),
    plot(theta,phi(l,:),'Color',clrs(l)*ones(1,3),'Linewidth',3);
end

% (do this separately from other for to get the right legends)
for l=1:length(lambda),
    plot(thetaStar(l),phiStar(l),'ko');
    
    % == Verify that thetaStar really maximizes phi ==
    [phimax,thetamax] = max(phi(l,:));
    plot(theta(thetamax),phimax,'rx');
end

legend({'$\lambda=1.1$','$\lambda=1.25$','$\lambda=1.5$','$\lambda=2$','$\lambda=5$'},'Location','northwest','Interpreter','latex','fontsize',15);
ylim([0,pi/6])
set(gca,'XTick',[0,pi/8,pi/4,3*pi/8,pi/2],'xticklabel',{'0','\pi/8','\pi/4','3\pi/4','\pi/2'},'fontsize',12);
set(gca,'YTick',[0,pi/24,pi/12,pi/8,pi/6],'yticklabel',{'0','\pi/24','\pi/12','\pi/8','\pi/6'},'fontsize',12);
ylabel('$\varphi$','Interpreter','latex','fontsize',25);
xlabel('$\theta$','Interpreter','latex','fontsize',25);
set(gcf,'PaperUnits','centimeters','PaperSize',[15,10],'PaperPosition',[0,0,15,10]);
set(gcf, 'renderer','default');
figurename = 'figure2.pdf';
saveas(gcf,figurename);



