% clear
% close all
% 
% addpath(genpath('./datasets'))
% addpath(genpath('./algorithms'))
% 
% %% Load the data
% 
% load('ICE_DATA.mat')
% 
% 
% %% Random sketching maps to reduce dimension
% 
% XC0 = cell(60,1);
% 
% pf = parfor_progress(60);
% pfcleanup = onCleanup(@() delete(pf));
% 
% for delays = 1:60
%     X = DATA(:,delays:end-1);
%     XC = randn(round(2*size(X,2)),size(X,1))*X; % random sketching to reduce dimension
% 
%     for jj = delays-1:(-1):1 % delay embedding
%         XC = XC + randn(round(2*size(X,2)),size(X,1))*DATA(:,jj:end-(delays-jj+1));
%     end
%     XC0{delays} = XC;
%     parfor_progress(pf);
% end
% clear X

%% Forecast errors

Err = zeros(9,12,6);
Pp = Err;
pf = parfor_progress(9*12*6);
pfcleanup = onCleanup(@() delete(pf));

for yr = 1:9
    for mt = 1:12
        for jj = 1:6
            
            delays = DE(yr,mt,jj);
            gp = GP(yr,mt,jj);
 
            X = DATA(:,delays:end-1);
            XC = XC0{delays};
            
            INDX = (yr+2011-1979)*12+mt-(delays-1);

            II = INDX-jj:(-1):1; II = II(1:gp*12+1); % indices of training data
        
            X2 = fliplr(XC(:,II)); % training data
            [~,~,L,PXs,PYs] = kernel_dictionaries(X2(:,1:end-1),X2(:,2:end),'type',"poly_exp");
            K = PXs\PYs;

            [W,LAM,W2] = eig(K,'vector');
    
            t1 = X(ACT{mt},INDX); % true classification  
            t1(t1<=15)=0; t1(t1>15)=1; % convert to binary problem
    
            R = (sqrt(real(diag(W2'*L*W2)./diag(W2'*W2)-abs(LAM).^2))); % errors of eigenpairs                
            [~,I] = sort(R,'ascend');
            nn = knee_pt(R(I(max(1,length(I)-110):end-10)))+max(1,length(I)-110)-1;

            PXr = PXs*W(:,I(1:nn)); PYr = PYs*W(:,I(1:nn));

            c = ([PXr(1,:);PYr])\transpose(fliplr(X(ACT{mt},II))); % coefficients in dictionary        
            y1 = transpose(real(PYr(end,:)*((LAM(I(1:nn)).^jj).*c))); % prediction
            y1(y1<=15)=0; y1(y1>15)=1;

            Err(yr,mt,jj) = sum(abs(y1-t1),1)/length(ACT{mt})*100;          
            parfor_progress(pf);
        end
       
    end
end
%%
Err(9,10:12,:,:,:,:)=NaN; % cut off data not included in IceNet paper

%% Plot the results

er1 = Err;
er0=squeeze(mean(er1,[1,2],"omitnan"));
er0=100-er0;

% icenet errors
ice_err = [96.9,96.4,96.4,96.4,96.3,96.3;
    96.9,96,95.8,95.8,95.7,95.7;
    96.9,95.7,95.3,95,95.1,95.1;
    97.1,95.9,95.4,95.3,95.1,95.1;
    97.5,96.6,96.3,96.0,95.7,95.6;
    96.0,94.5,94.1,94.1,93.9,93.7;
    94.2,91.7,90.7,90.6,90.9,90.5;
    94,92.1,91,90.5,90.5,90.2;
    94.3,92.9,92.2,91.1,90.4,90.4;
    93,92.4,92,91.8,90.7,89.8;
    95.4,95.3,94.6,94.6,94.7,94.8;
    96.9,96.4,96.3,96.2,96.3,96.3];


f1 = [-0.1,0.8,0.8,0.8,0.8,0.8;
    -0.2,0.1,0.4,0.3,0.6,0.6;
    -0.5,-0.1,-0.1,0.2,0.3,0.4;
    -0.6,0.2,0.3,0.6,0.7,0.6;
    -0.3,0.3,0.7,0.7,0.9,0.9;
    -0.5,0.3,0.1,0,0.2,-0.2;
    -0.9,-0.4,0.9,0.2,0.6,0.6;
    -1.2,0.5,1.9,2.1,1.2,1.0;
    -1.9,1.1,2.2,2.7,2.9,2.6;
    2.4,2.2,2.4,2.5,0.7,-0.3;
    -0.4,1.1,0.3,0.8,0.6,0.6;
    0,-0.2,0.3,0.3,0.5,0.5];

sea_err = ice_err-f1;

%% Plot the results

figure
plot(mean(ice_err,1),'.-','markersize',25,'linewidth',3)
hold on
plot(mean(sea_err,1),'.:','markersize',25,'linewidth',3)
plot(er0,'x-','markersize',17,'linewidth',3)
ylim([92.9,96.2])
grid on


xlabel('Lead Time (Months)','interpreter','latex','fontsize',14)
ylabel('Binary Accuracy','interpreter','latex','fontsize',14)
legend({'IceNet','SEAS5','Proposed Method'},'interpreter','latex','fontsize',12,'location','north')

exportgraphics(gcf,'sea_ice_forecast2.pdf');

%%
% C = linspecer(6);
bb = 100-(squeeze(mean(er1,1,'omitnan')));
% close all
figure
plot(1:12,mean((bb(:,:)-ice_err(:,:)),2),'.-','linewidth',3,'markersize',25)
hold on
plot(1:12,mean((bb(:,:)-sea_err(:,:)),2),'.:','linewidth',3,'markersize',25)
plot(1:12,zeros(1,12),':','linewidth',2,'color',[1,1,1]*0.5)
xlim([0.5,12.5])
ylim([-0.1,2.5])
xlabel('Month','interpreter','latex','fontsize',14)
title({'Proposed Method''s','Binary Accuracy Improvement'},'interpreter','latex','fontsize',14)
legend({'Compared to IceNet','Compared to SEAS5'},'interpreter','latex','fontsize',12,'location','best')
grid on
exportgraphics(gcf,'sea_ice_forecast3.pdf');


figure
plot(1:12,mean((bb(:,:)),2),'x-','color','k','linewidth',3,'markersize',25)
xlim([0.5,12.5])
xlabel('Month','interpreter','latex','fontsize',14)
title({'Proposed Method''s','Mean Binary Accuracy'},'interpreter','latex','fontsize',14)
grid on
exportgraphics(gcf,'sea_ice_forecast4.pdf');
