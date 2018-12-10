clear all; clc;

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
                        0.0000 0.0092 0.9836 0.0082 0.000;
                        0.0000 0.0000 0.0153 0.9806 0.0041;
                        0.0000 0.0000 0.0000 0.0273 0.9727];
aalpha.values       = [0.25,0.3,0.35];
aalpha.transition   = [0.9 0.07 0.03; 0.05 0.9 0.05; 0.03 0.07 0.9];
tol                 = 1e-6;
productionss        = 100;

nz                  = length(z.values);
naalpha             = length(aalpha.values);
nCapitalGrid        = [100 500 5000];

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

%% Multigrid
kmin = (1-0.3) * kss;
kmax = (1+0.3) * kss;
mTransition = kron(aalpha.transition,z.transition);
option = optimoptions('fsolve','Display','none','FunctionTolerance',1e-7);
V = uss*ones(nCapitalGrid(1),naalpha,nz);
iter_Multigrid = zeros(1,length(nCapitalGrid));
time_Multigrid = zeros(1,length(nCapitalGrid));

policy.k = zeros(nk,naalpha,nz);
policy.l = zeros(nk,naalpha,nz);
policy.c = zeros(nk,naalpha,nz);

for i = 1 : length(nCapitalGrid)
    tic;
    
    nk = nCapitalGrid(i);
    grid.capital = linspace(kmin,kmax,nk);    
    
    diff = inf;
    iter_Multigrid(i) = 1;
    while diff > tol
        val = @(k) value_function(k,V,option);

        tempV = zeros(nk,naalpha,nz);
    
        for ind = 1 : nz*naalpha*nk
            ik = floor(mod(ind-0.05,nk))+1;
            iz = mod(floor((ind-0.05)/nk),nz)+1;
            iaalpha = mod(floor((ind-0.05)/(nk*nz)),naalpha)+1;
        
            capital = grid.capital(ik);
            tfp = z.values(iz);
            capitalElasticity = aalpha.values(iaalpha);

            capitalNext = fminbnd(val,kmin,kmax);
            policy.k(ik,iaalpha,iz) = capitalNext;
            policy.i(ik,iaalpha,iz) = investment;
            policy.c(ik,iaalpha,iz) = consumption;
            policy.l(ik,iaalpha,iz) = labor;

            tempV(ik,iaalpha,iz) = -val(capitalNext);
        end
        diff = abs(tempV-V);
        diff = max(diff(:));    
        V = tempV;

        if (mod(iter_Multigrid(i),10)==0 || iter_Multigrid(i) ==1)
            fprintf(' Iteration = %d, Sup Diff = %2.8f\n', iter_Multigrid(i), diff); 
        end
        iter_Multigrid(i) = iter_Multigrid(i) + 1;
    end
    time_Multigrid(i) = toc;
    
    if i ~= length(nCapitalGrid)
        grid.capitalOld = grid.capital;
        grid.capital = linspace(kmin,kmax,nCapitalGrid(i+1));
        nk = length(grid.capital);
        Vold = V;
        V = zeros(nk,naalpha,nz);
        policy.k = zeros(nk,naalpha,nz);
        policy.l = zeros(nk,naalpha,nz);
        policy.c = zeros(nk,naalpha,nz);
        
        V(1,:,:) = Vold(1,:,:);
        V(end,:,:) = Vold(end,:,:);
        for ik = 2 : nk - 1
            capital = grid.capital(ik);
            for iaalpha = 1 : naalpha
                for iz = 1 : nz
                    V(ik,iaalpha,iz) = interpolate(Vold(:,iaalpha,iz),capital,grid.capitalOld);
                end
            end
        end
        
    end
    
end

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
    saveas(gcf,['..' filesep 'Output' filesep 'Q6_Multigrid_' file_label{i} '.png']);
    close(gcf)
end

%% Plot impulse response functions
% Case 1: z=0 --> z=0.0673
shock_variable = {'z','alpha'};

T = 60;
cPath = [policy.c(2525,2,3) zeros(1,T-1)];
cPath_percentdev = zeros(1,T);
kPath = [policy.k(2525,2,3), zeros(1,T-1)];
kPath_percentdev = zeros(1,T);
lPath = [policy.l(2525,2,3) zeros(1,T-1)];
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
        cPath_percentdev(t) = (log(cPath(t))-log(policy.c(2525,2,3)))*100;

        kPath(t) = interpolate(policy.k(:,aPath(t),zPath(t)),kPath(t-1),grid.capital);
        kPath_percentdev(t) = (log(kPath(t))-log(policy.k(2525,2,3)))*100;

        lPath(t) = interpolate(policy.l(:,aPath(t),zPath(t)),kPath(t-1),grid.capital);
        lPath_percentdev(t) = (log(lPath(t))-log(policy.l(2525,2,3)))*100;
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
    
    saveas(gcf,['..' filesep 'Output' filesep 'Q6_Multigrid_IRF_' shock_variable{i} '.png']);
    close(gcf)
end

%% Compute Euler Equation Errors
[euler_error_Multigrid,max_error_Multigrid] = euler_error_VFI(aalpha,bbeta,ggamma,ddelta,ppsi, ...
    eeta,V,policy.c,policy.l,policy.k,z,grid.capital,naalpha,nz,nk);

%% Save results
save(['..' filesep 'Output' filesep 'Q6_Multigrid.mat'])