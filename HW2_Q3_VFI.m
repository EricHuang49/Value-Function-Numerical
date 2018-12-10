% This file uses value function iteration to calculate value and policy
% functions for an RBC model. The policy functions are generated using
% linear interpolation.

clear all; clc;
addpath('Functions')

%% Set up parameters
global bbeta ggamma ddelta ppsi z aalpha eeta nz naalpha lss iz iaalpha ...
        capital tfp capitalElasticity labor grid investment consumption nk

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

nz                  = length(z.values);
naalpha             = length(aalpha.values);
nkList              = 250;                      % can start with fewer cpaital grid points to form better initial guess

%% Calculate the steady state
zss = z.values(3);
aalphass = aalpha.values(2);
kl = ((1-bbeta*(1-ddelta))/(bbeta*aalphass)*exp(zss))^(1/(aalphass-1));     % k/l
lss = productionss/(exp(zss)*kl^aalphass);                                  % steady state labor
css = (exp(zss)*kl^aalphass-ddelta*kl)*lss;                                 % steady state consumption
kss = lss * kl;                                                             % steady state capital
eeta = (1-aalphass)*exp(zss)*kl^aalphass/(lss*css);                         % parameter for labor disutility
uss = (log(css)-eeta*lss^2/2);                                              % utility at steady state
clear zss aalphass kl

%% Generate grid for capital
kmin = (1-0.3) * kss;
kmax = (1+0.3) * kss;

%% VFI with linear interpolation
option = optimoptions('fsolve','Display','none','FunctionTolerance',1e-7);

tic
for i = 1 : length(nkList)
    nk = nkList(i);
    fprintf(' Now running with %d grid points\n',nk);
    grid.capital = linspace(kmin,kmax,nk);
    
    policy.k = zeros(nk,naalpha,nz);
    policy.i = zeros(nk,naalpha,nz);
    policy.c = zeros(nk,naalpha,nz);
    policy.l = zeros(nk,naalpha,nz);
    
    if i == 1
        V = uss*ones(nk,naalpha,nz);
    else        % interpolate the previous value function on the new grid to get an initial guess
        for ik = 1 : nk
            for iaalpha = 1 : naalpha
                for iz = 1 : nz
                    V(ik,iaalpha,iz) = interpolate(Vold(:,iaalpha,iz),grid.capital(ik),grid.capitalOld);
                end
            end
        end
    end

    diff = inf;
    iter_VFI = 1;
    while diff > tol
        val = @(k) value_function(k,V,option);
        tempV = zeros(nk,naalpha,nz);

        for ik = 1 : nk
            capital = grid.capital(ik);
            for iaalpha = 1 : naalpha
                capitalElasticity = aalpha.values(iaalpha);
                for iz = 1 : nz
                    tfp = z.values(iz);

                    capitalNext = fminbnd(val,kmin,kmax);
                    policy.k(ik,iaalpha,iz) = capitalNext;
                    policy.i(ik,iaalpha,iz) = investment;
                    policy.c(ik,iaalpha,iz) = consumption;
                    policy.l(ik,iaalpha,iz) = labor;

                    tempV(ik,iaalpha,iz) = -val(capitalNext);
                end
            end
        end
        diff = abs(tempV-V);
        diff = max(diff(:));  
        V = tempV;

        if (mod(iter_VFI,10)==0 || iter_VFI ==1)
            fprintf(' Iteration = %d, Sup Diff = %2.8f\n', iter_VFI, diff); 
        end
        iter_VFI = iter_VFI + 1;
    end
    Vold = V;
    grid.capitalOld = grid.capital;
    
end
time_VFI = toc;


%% Plot the functions
v_fun = reshape(V,nk,naalpha*nz);
g_k = reshape(policy.k,nk,naalpha*nz);
g_c = reshape(policy.c,nk,naalpha*nz);
g_l = reshape(policy.l,nk,naalpha*nz);

series_names = {'v_fun','g_c','g_k','g_l'};
title_names = {'Value Function','Consumption Decision','Capital Next Period','Labor Decision'};
ylabel_names = {'Value','Consumption','Capital tomorrow','Labor'};
ylims = {[3.73 3.85],[55 90],[190 360],[60 70]};
file_label = {'V','C','K','L'};
num_plots = length(series_names);

for i = 1 : num_plots
    subplot(2,1,1)
    plot_function = ['plot(grid.capital,' series_names{i} '(:,[2 5 8 11 14]))'];
    evalin('base',plot_function);
    title([title_names{i} ', \alpha=0.3'])
    legend('z = -0.0673','z = -0.0336','z = 0','z = 0.0336','z = 0.0673')
    xlabel('Capital Today','FontSize',7)
    ylabel(ylabel_names{i},'FontSize',7)
    ylim(ylims{i})
    
    subplot(2,1,2)
    plot_function = ['plot(grid.capital,' series_names{i} '(:,7:9))'];
    evalin('base',plot_function);
    title([title_names{i} ', z=0'])
    legend('\alpha = 0.25','\alpha = 0.3','\alpha = 0.35');
    xlabel('Capital Today','FontSize',7)
    ylabel(ylabel_names{i},'FontSize',7)
    ylim(ylims{i})
    saveas(gcf,['..' filesep 'Output' filesep 'Q3_VFI_' file_label{i} '.png']);
    close(gcf)
end

%% Plot impulse response functions
% Case 1: z=0 --> z=0.0673
shock_variable = {'z','alpha'};

T = 60;
cPath = [policy.c(126,2,3) zeros(1,T-1)];
cPath_percentdev = zeros(1,T);
kPath = [policy.k(126,2,3), zeros(1,T-1)];
kPath_percentdev = zeros(1,T);
lPath = [policy.l(126,2,3) zeros(1,T-1)];
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
        cPath(t) = interpolate(policy.c(:,aPath(t),zPath(t)),kPath(t-1),grid.capital);
        cPath_percentdev(t) = (log(cPath(t))-log(policy.c(126,2,3)))*100;

        kPath(t) = interpolate(policy.k(:,aPath(t),zPath(t)),kPath(t-1),grid.capital);
        kPath_percentdev(t) = (log(kPath(t))-log(policy.k(126,2,3)))*100;

        lPath(t) = interpolate(policy.l(:,aPath(t),zPath(t)),kPath(t-1),grid.capital);
        lPath_percentdev(t) = (log(lPath(t))-log(policy.l(126,2,3)))*100;
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
    
    saveas(gcf,['..' filesep 'Output' filesep 'Q3_VFI_IRF_' shock_variable{i} '.png']);
    close(gcf)
end

%% Compute Euler Equation Errors
[euler_error_VFI,max_error_VFI] = euler_error_VFI(aalpha,bbeta,ggamma,ddelta,ppsi, ...
    eeta,V,policy.c,policy.l,policy.k,z,grid.capital,naalpha,nz,nk);

%% Save results
save(['..' filesep 'Output' filesep 'Q3_VFI.mat'])