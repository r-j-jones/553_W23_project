function [fig] = DisplayInputData(M, L, S, W, params)

fig = figure;
set(fig,'Position',[1779 -25 1517 269],'Color','w','InvertHardcopy','off');

subplot(141);
imagesc(M); 
% imshow(M,Border="tight");
titlestr = [' M (data) = '];
title(titlestr);
set(gca,'CLim',[-1 1],'Colormap',gray);
colorbar; drawnow;

subplot(142);
imagesc(L); 
titlestr = [' L (lowrank, r=' num2str(params.r) ') + '];
title(titlestr);
set(gca,'CLim',[-1 1],'Colormap',gray);
colorbar; drawnow;

subplot(143);
imagesc(S); title(' S (sparse) + ')
titlestr = [' S (sparse, p=' sprintf('%g',params.p) ') + '];
title(titlestr);
set(gca,'CLim',[-1 1],'Colormap',gray);
colorbar; drawnow;

subplot(144);
imagesc(W); 
titlestr = [' W (noise, \sigma=' sprintf('%g',params.sigman) ') '];
title(titlestr,'Interpreter','tex');
set(gca,'CLim',[-(params.sigman^2) params.sigman^2],'Colormap',gray);
colorbar; drawnow;

axs = findall(gcf,'Type','Axes');
set(axs,'FontSize',14)