clear all; clc;
addpath('Functions')

%% Set up parameters
bbeta               = 0.99;
ggamma              = 10;
ddelta              = 0.1;
ppsi                = 0.5;
z.values            = [-0.0673, -0.0336,0,0.0336,0.0673];
z.transition        = [0.9727 0.0273 0.0000 0.0000 0.0000;
                        0.0041 0.9806 0.0153 0.0000 0.0000;
                        0.0000 0.0092 0.9836 0.0082 0.0000;
                        0.0000 0.0000 0.0153 0.9806 0.0041;
                        0.0000 0.0000 0.0000 0.0273 0.9727];
aalpha.values       = [0.25,0.3,0.35];
aalpha.transition   = [0.9 0.07 0.03; 0.05 0.9 0.05; 0.03 0.07 0.9];
tol                 = 1e-6;
productionss        = 100;
numNodeList         = [3 4 5 6 7 8];

nz                  = length(z.values);
naalpha             = length(aalpha.values);
nk                  = 250;
T_sim               = 10000;
dropT               = 1000;

%% Calculate the steady state
zss = z.values(3);
aalphass = aalpha.values(2);
kl = ((1-bbeta*(1-ddelta))/(bbeta*aalphass)*exp(zss))^(1/(aalphass-1));
lss = productionss/(exp(zss)*kl^aalphass);
css = (exp(zss)*kl^aalphass-ddelta*kl)*lss;
kss = lss * kl;
eeta = (1-aalphass)*exp(zss)*kl^aalphass/(lss*css);
uss = (log(css)-eeta*lss^2/2);
clear zss aalphass kl

kmin = (1-0.3) * kss;
kmax = (1+0.5) * kss;

%% Solve the model using Chebyshev polynomials
tic;
for step = 1 : length(numNodeList)

    numNode = numNodeList(step);
    M = numNode*nz*naalpha;

    % Find Zeros of the Chebychev Polynomial on order M 
    ZC = -cos((2*(1:numNode)'-1)*pi/(2*numNode));

    % Define Chebychev polynomials
    T_k = ones(numNode,numNode);
    T_k(:,2) = ZC;

    for i1 = 3 : numNode
        T_k(:,i1) = 2*ZC.*T_k(:,i1-1) - T_k(:,i1-2);
    end

    % Project collocation points in the K space
    grid.k = ((ZC+1)*(kmax-kmin))/2+kmin;

    % Initial Guess for Chebyshev coefficients
    coeffGuess = zeros(2*M,1);
    
    if step == 1
        for iaalpha = 1 : naalpha
            for iz =  1: nz
                ind = (iaalpha-1)*nz + iz;
                coeffGuess((ind-1)*numNode+1) = uss;
                coeffGuess((ind-1)*numNode+M+1) = lss;
            end
        end
    else
        for iaalpha = 1 : naalpha
            for iz = 1 : nz
                ind = (iaalpha-1)*nz + iz;
                coeffGuess((ind-1)*numNode+1:(ind-1)*numNode+numNodeOld) = ...
                    coeffGuessOld((ind-1)*numNodeOld+1:ind*numNodeOld);
                coeffGuess((ind-1)*numNode+M+1:(ind-1)*numNode+M+numNodeOld) = ...
                    coeffGuessOld((ind-1)*numNodeOld+MOld+1:ind*numNodeOld+MOld);
            end
        end
    end

    % Solve for Chebyshev coefficients
    coefficients = residual_function(aalpha,bbeta,ggamma,ddelta,ppsi,eeta,kmin,kmax,coeffGuess,grid.k,T_k,z,numNode,naalpha,nz,M);
    coeffGuessOld = coefficients;
    numNodeOld = numNode;
    MOld = M;
end
time_Chebyshev = toc;

%% Compute Euler Errors
grid.k_complete = linspace(kmin,kmax,nk)';

[g_k,g_c,g_l,value_fcn,euler_error_Chebyshev,max_error_Chebyshev]= ...
                    eulerr_grid(aalpha,bbeta,ggamma,ddelta,ppsi,eeta,coefficients,z,...
                    kmin,kmax,grid.k_complete,naalpha,nz,numNode,nk,M);

%% Figures
% Decision Rules
series_names = {'value_fcn','g_c','g_k','g_l'};
title_names = {'Value Function','Consumption Decision','Capital Next Period','Labor Decision'};
ylabel_names = {'Value','Consumption','Capital tomorrow','Labor'};
ylims = {[3.73 3.85],[55 90],[190 360],[60 70]};
file_label = {'V','C','K','L'};
num_plots = length(series_names);

for i = 1 : num_plots
    subplot(2,1,1)
    plot_function = ['plot(grid.k_complete,' series_names{i} '(:,[6 7 8 9 10]))'];
    evalin('base',plot_function);
    title([title_names{i} ', \alpha=0.3'])
    legend('z = -0.0673','z = -0.0336','z = 0','z = 0.0336','z = 0.0673')
    xlabel('Capital Today','FontSize',7)
    ylabel(ylabel_names{i},'FontSize',7)
    ylim(ylims{i}); xlim([180 360]);
    
    subplot(2,1,2)
    plot_function = ['plot(grid.k_complete,' series_names{i} '(:,[3 8 13]))'];
    evalin('base',plot_function);
    title([title_names{i} ', z=0'])
    legend('\alpha = 0.25','\alpha = 0.3','\alpha = 0.35');
    xlabel('Capital Today','FontSize',7)
    ylabel(ylabel_names{i},'FontSize',7)
    ylim(ylims{i}); xlim([180 360]);
    saveas(gcf,['..' filesep 'Output' filesep 'Q7_Chebyshev_' file_label{i} '.png']);
    close(gcf)
end

% Euler Equation Error on the Grid
figure(2)
plot(grid.k_complete,euler_error_Chebyshev)
title('Log10 Euler Error')
                
%% Simulation
[vSeries,kSeries,cSeries,lSeries] = ...
    simulation(aalpha,bbeta,ggamma,ddelta,ppsi,eeta,kss,coefficients,z,kmin,kmax,numNode,naalpha,nz,M,T_sim,dropT);

%% IRF
shock_variable = {'z','alpha'};

T = 60;
cPath = [g_c(95,8) zeros(1,T-1)];
cPath_percentdev = zeros(1,T);
kPath = [g_k(95,8), zeros(1,T-1)];
kPath_percentdev = zeros(1,T);
lPath = [g_l(95,8) zeros(1,T)];
lPath_percentdev = zeros(1,T);

for i = 1 : length(shock_variable)
    if i == 1
        zPath = [3 4 ones(1,T-2)*3];
        aPath = 2*ones(1,T);
    else
        zPath = 3*ones(1,T);
        aPath = [2 3 ones(1,T-2)*2];
    end
    
    for t = 2 : T
        cPath(t) = interpolate(g_c(:,(aPath(t)-1)*nz+zPath(t)),kPath(t-1),grid.k_complete);
        cPath_percentdev(t) = (log(cPath(t))-log(g_c(95,8)))*100;
        
        kPath(t) = interpolate(g_k(:,(aPath(t)-1)*nz+zPath(t)),kPath(t-1),grid.k_complete);
        kPath_percentdev(t) = (log(kPath(t))-log(g_k(95,8)))*100;
        
        lPath(t) = interpolate(g_l(:,(aPath(t)-1)*nz+zPath(t)),kPath(t-1),grid.k_complete);
        lPath_percentdev(t) = (log(lPath(t))-log(g_l(95,8)))*100;
    end
    
    subplot(2,2,1)
    plot(1:T,cPath_percentdev,1:T,zeros(1,T),':')
    title(['% deviation for c, shock to ' shock_variable{i}])
    xlabel('period','FontSize',7)
    subplot(2,2,2)
    plot(1:T,kPath_percentdev,1:T,zeros(1,T),':')
    title(['% deviation for k, shock to ' shock_variable{i}])
    xlabel('period','FontSize',7)
    subplot(2,2,3)
    plot(1:T,lPath_percentdev,1:T,zeros(1,T),':')
    title(['% deviation for l, shock to ' shock_variable{i}])
    xlabel('period','FontSize',7)
    
    saveas(gcf,['..' filesep 'Output' filesep 'Q7_Chebyshev_IRF_' shock_variable{i} '.png']);
    close(gcf)
end

%% Save results
save(['..' filesep 'Output' filesep 'Q7_Chebyshev.mat'])