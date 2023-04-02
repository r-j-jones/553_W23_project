% ====================================================================
% Sample code to replicate the noiseless experiments in
%
%   D. Pimentel-Alarcon, R. Nowak
%   Random Consensus Robust PCA,
%   International Conference on Artificial Intelligence and Statistics
%   (AISTATS), 2017.
%
% This code generates low-rank plus sparse matrices with varying
% coherences and sparsity levels, and then runs R2PCA to decompose
% these matrices (see Figure 2).
%
% Written by: D. Pimentel-Alarcon.
% email: pimentelalar@wisc.edu
% Created: 2017
% =====================================================================

clear all; close all; warning ('off','all'); clc;
rand('state',sum(100*clock));

% ======================================================================
% =========================== GENERAL SETUP ============================
% ======================================================================
d = 100;        % Size of the matrix = d x N
N = d;
r = 5;          % rank of low-rank component
% P = 0.01; %.01:.04:.17;% Probability of nonzero entry in sparse component
p = 0.01;
% Mu = 2; %2:4:19;    % Coherence: value in [1,d/r]
mu = 2;
v = 10;         % variance of the sparse entries %v=10 used in paper
T = 1;          % Number of trials
k = 10; %2*r;
NoiseSigma = 1e-6;

% r = 5;          % rank of low-rank component
% P = .01:.01:.17;% Probability of nonzero entry in sparse component
% Mu = 2:1:19;    % Coherence: value in [1,d/r]
% v = 10;         % variance of the sparse entries
% T = 10;   


fprintf('d=%d\tN=%d\tr=%g\tp=%g\nmu=%g\tv=%g\tk=%g\tsigman=%g\n', d,N,r,p,mu,v,k,NoiseSigma);
% [err,Time,mu] = runExperiment(d,N,r,P(p),v,Mu(mu));

[err,Time,mu] = runExperiment_Noise(d, N, r, p, v, mu, NoiseSigma,k);
            

% ===== Auxiliary variables to keep track of errors =====
Err = -ones(length(Mu),length(P),T);
Time = -ones(length(Mu),length(P),T);
MU = -ones(length(Mu),length(P),T);

% ======================================================================
% ========================== Run experiments ===========================
% ======================================================================
for t=1:T,
    for mu=1:length(Mu);
        for p=1:length(P),
            fprintf('d = %d, N = %d, r = %d, p = %1.1d,  v = %d,  trial = %d  ',d,N,r,P(p),v,t);
            [Err(mu,p,t),Time(mu,p,t),MU(mu,p,t)] = runExperiment(d,N,r,P(p),v,Mu(mu));
            fprintf('error = %1.1d, time = %1.1d. \n',Err(mu,p,t),Time(mu,p,t));
        end
    end
end

% ======================================================================
% ===================== Plot transition diagrams =======================
% ======================================================================

% ====================== Error ======================
figure(1);
imagesc(1-mean(Err,3));
colormap(gray);
set(gca,'YDir','normal');
title('R2PCA Success Rate');
ylabel('$\mu$','Interpreter','latex','fontsize',30);
xlabel('$p$','Interpreter','latex','fontsize',30);

set(gca,'XTick',1:2:length(P),'xticklabel',min(P):.02:max(P),'fontsize',10);
set(gca,'YTick',1:2:length(Mu),'yticklabel',min(Mu):2:max(Mu),'fontsize',10);


% ====================== Time ======================
figure(2);
imagesc(1-mean(Time,3));
colormap(gray);
set(gca,'YDir','normal');
title('R2PCA Time');
ylabel('$\mu$','Interpreter','latex','fontsize',30);
xlabel('$p$','Interpreter','latex','fontsize',30);

set(gca,'XTick',1:2:length(P),'xticklabel',min(P):.02:max(P),'fontsize',10);
set(gca,'YTick',1:2:length(Mu),'yticklabel',min(Mu):2:max(Mu),'fontsize',10);




