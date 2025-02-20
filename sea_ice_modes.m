clear
close all
addpath(genpath('./algorithms'))
addpath(genpath('./data_sets'))


%% Load the data

load('ICE_DATA.mat')
delays = 6; % number of time delays, 5 for five year forecast
X = DATA(:,delays:end-1);
for jj = delays-1:(-1):1
    X = [X;DATA(:,jj:end-(delays-jj+1))];
end


%% Run the algorithm

[~,K,L,PX,PY] = kernel_dictionaries(X(:,1:end-1),X(:,2:end),'type',"Gaussian");
[W,LAM,W2] = eig(K,'vector');

R = (sqrt(real(diag(W2'*L*W2)./diag(W2'*W2)-abs(LAM).^2))); % error bounds
[~,I] = sort(R,'ascend');
W = W(:,I); LAM = LAM(I); W2 = W2(:,I); R = R(I);

PXr = PX*W; PYr = PY*W;

%% Error bounds of EDMD eigenvalues

[~,I] = sort(R,'descend');
W = W(:,I); LAM = LAM(I)/max(abs(LAM)); W2 = W2(:,I); R = R(I);

figure
n2 = length(LAM);
scatter(angle(LAM(end-n2+1:end)),log(abs(LAM(end-n2+1:end))),1000*R(end-n2+1:end),R(end-n2+1:end),'filled','MarkerEdgeColor','k','LineWidth',0.01);
n2 = 17;
hold on
scatter(angle(LAM(end-n2+1:end)),log(abs(LAM(end-n2+1:end))),1000*R(end-n2+1:end),'b','filled','MarkerEdgeColor','y','LineWidth',0.02);
set(gca, 'GridColor', 'b')
hold on
load('cmap.mat')
colormap(cmap2); colorbar
xlabel('$\mathrm{arg}(\lambda)$','interpreter','latex','fontsize',18)
ylabel('$\log(|\lambda|)$','interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=18;
xlim([-pi,pi])
ylim([-0.015,0.002])
clim([0,0.15])
grid on
box on
% exportgraphics(gcf,'sea_ice_gauss_errors.png','Resolution',300);
return
%% Koopman modes

[~,I] = sort(R,'ascend');
W = W(:,I); LAM = LAM(I); W2 = W2(:,I); R = R(I);
Phi = transpose((PX*W)\(X(1:97877,1:end-1)'));

c = vecnorm(Phi);
d = vecnorm(c.*(abs(LAM).^(0:100000))');
d = d.^2/(97877*504);
d = 1./d;

for jj =1:17
    figure
    u = real(Phi(1:size(DATA,1),jj));
    u = real(u*exp(1i*mean(angle(u))));
    
    v = zeros(432*432,1)+NaN;
    v(nLAND) = u(:);
    v = reshape(v,[432,432]);
    imagesc(v,'AlphaData',~isnan(v))
    colormap(coolwarm)
    set(gca,'Color',[1,1,1]*0.4)
    clim([-std(u),std(u)]*4+mean(u));
    title(sprintf('$|\\lambda|=%.3f,\\mathrm{arg}(\\lambda)/\\pi=%.3f,r=%.3f$',abs(LAM(jj)/max(abs(LAM))),angle(LAM(jj))/pi,R(jj)),'interpreter','latex','fontsize',15)
    axis equal
    axis tight
    grid on
    set(gca,'xticklabel',{[]})
    set(gca,'yticklabel',{[]})
    exportgraphics(gcf,sprintf('sea_ice_gauss_%d.png',jj),'Resolution',300);

    pause(0.1)
    close all
end


%% Forecast experiment for five year forecast
clear
close all

load('ICE_DATA.mat')
delays = 5; % number of time delays, 5 for five year forecast
X = DATA(:,delays:end-1);
for jj = delays-1:(-1):1
    X = [X;DATA(:,jj:end-(delays-jj+1))];
end

mt = 1; % starting month
yr = 2015; % starting year
INDX = (yr-1979)*12+mt-delays; M = INDX-1;
real_data = X(1:97877,INDX+(1:12*5)); 
I2 = M-11*12+1:M;

%% error bound approach

[~,K,L,PXs,PYs] = kernel_dictionaries(X(:,I2),X(:,I2+1),'type',"Gaussian");
[W,LAM,W2] = eig(K,'vector');
R = (sqrt(real(diag(W2'*L*W2)./diag(W2'*W2)-abs(LAM).^2)));
[~,I] = sort(R,'ascend');
N = knee_pt(R(I(max(1,length(I)-40):end-10)))+max(1,length(I)-40)-1;
PXr = PXs*W(:,I(1:N)); PYr = PYs*W(:,I(1:N));
    
c = ([PXr(1,:);PYr])\transpose(X(1:97877,[I2,M+1]));
y1 = real(transpose(transpose(PYr(end,:)).*(LAM(I(1:N)).^(1:12*5)))*c)';

%% Compare to DMD

[U,S,~] = svd(X(:,1:M),'econ');
r = rank(S);
U = U(:,1:r);
            
PXs = X(:,1:M)'*U;
PYs = X(:,2:M+1)'*U;
K = PXs\PYs;
[W,LAM,W2] = eig(K,'vector');
PXr = PXs*W; PYr = PYs*W;

c = ([PXr(1,:);PYr])\transpose(X(1:97877,1:M+1));
y2 = real(transpose(transpose(PYr(end,:)).*(LAM.^(1:12*5)))*c)';

%% Plot the results
er1 = sum(abs(y1-real_data).^2,1)./sum(abs(real_data).^2,1);
er2 = sum(abs(y2-real_data).^2,1)./sum(abs(real_data).^2,1);
figure
plot(er2,'m:','linewidth',2)
hold on
plot(er1,'g:','linewidth',2)

b1 = movmean(er1,12);
b1(1:6)="NaN"; b1(end-5:end)="NaN"; % ignore endpoints
b2 = movmean(er2,12);
b2(1:6)="NaN"; b2(end-5:end)="NaN"; % ignore endpoints

semilogy(b2,'m','linewidth',3)
semilogy(b1,'g','linewidth',3)
grid on

xlabel('Lead Time (Months)','interpreter','latex','fontsize',14)
ylabel('Forecast Error','interpreter','latex','fontsize',14)
legend({'DMD','Proposed Method'},'interpreter','latex','fontsize',12,'location','best')
% exportgraphics(gcf,'sea_ice_forecast1.pdf');

%% Plot the sea ice extent

clear
close all
addpath(genpath('./algorithms'))
addpath(genpath('./data_sets'))

load('ICE_DATA.mat')
delays = 6; % number of time delays
X = DATA(:,delays:end-1);
for jj = delays-1:(-1):1
    X = [X;DATA(:,jj:end-(delays-jj+1))];
end

% Run the algorithm for 10 year blocks
for block = 1:4
    It = (1:12*10) + (block-1)*10*12 +6;
    
    [~,K,L,PX,PY] = kernel_dictionaries(X(:,It),X(:,It+1),'type',"Gaussian");
    
    [W,LAM,W2] = eig(K,'vector');
    
    R = (sqrt(real(diag(W2'*L*W2)./diag(W2'*W2)-abs(LAM).^2))); % dual residual
    [~,I] = sort(R,'ascend');
    W = W(:,I); LAM = LAM(I); W2 = W2(:,I); R = R(I);
    
    PXr = PX*W; PYr = PY*W;
    Phi = transpose((PX*W)\(X(1:97877,It)'));
    R(1:2)
    
    for jj =1:2
        figure
        u = real(Phi(1:size(DATA,1),jj));
        u = abs(u*exp(1i*mean(angle(u))));
        
        v = zeros(432*432,1)+NaN;
        v(nLAND) = u(:);
        v = reshape(v,[432,432]);
        imagesc(v,'AlphaData',~isnan(v))
        colormap(brighten(coolwarm,0.1))
        % colormap default
        set(gca,'Color',[1,1,1]*0.5)
        clim([0,max(u)*1]);
        axis equal
        axis tight
        grid on
        set(gca,'xticklabel',{[]})
        set(gca,'yticklabel',{[]})
        % exportgraphics(gcf,sprintf('sea_ice_block_%d_%d.png',block,jj),'Resolution',300);
    end
end

t = DATA; clear DATA
t(t<=15)=0; t(t>15)=1;
t=sum(t,1)*625;

f=figure;
plot(t,'linewidth',2)
hold on
t2=movmean(t,12);
t2(1:6)="NaN"; t2(end-5:end)="NaN";
plot(t2,'r','linewidth',3)

xticks(13:12*5:600)
xticklabels(1980:5:2030)
xlim([1,length(t)])
title('Sea Ice Extent','interpreter','latex','fontsize',14)
ylabel('km$^2$','interpreter','latex','fontsize',14)
grid on
f.Position=[360  160  700.0000  350];

% exportgraphics(gcf,'sea_ice_extent.png','Resolution',300);




