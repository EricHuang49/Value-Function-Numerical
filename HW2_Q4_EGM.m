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
                        0.0000 0.0092 0.9836 0.0082 0.000;
                        0.0000 0.0000 0.0153 0.9806 0.0041;
                        0.0000 0.0000 0.0000 0.0273 0.9727];
aalpha.values       = [0.25,0.3,0.35];
aalpha.transition   = [0.9 0.07 0.03; 0.05 0.9 0.05; 0.03 0.07 0.9];
tol                 = 1e-6;
productionss        = 100;

nz                  = length(z.values);
naalpha             = length(aalpha.values);
nk                  = 250;

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

%% Generate grid for capital
kmin = (1-0.3) * kss;
kmax = (1+0.5) * kss;
grid.capital = linspace(kmin,kmax,nk);

%% VFI with an Endogenous Grid:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 1: Endogenous Grid with labor fixed at the steady state
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

grid.Y = zeros(nk,naalpha,nz);                          % grid for Y generated from K_t+1
grid.Yend = zeros(nk,naalpha,nz);                       % endogenous grid for Y
grid.Vend = zeros(nk,naalpha,nz);                       % endogenous grid for V
grid.Vtilde = zeros(nk,naalpha,nz);                     % grid for V tilde
grid.Vupdated = zeros(nk,naalpha,nz);                   % grid for updated V, interpolated on K_t+1 grid
grid.VtildeUpdated = zeros(nk,naalpha,nz);              % grid for updated V tilde, calculated from Vupdated

diff = inf;
mTransition = kron(z.transition,aalpha.transition);
slope = linspace(1-0.0005*nk,1+0.0005*nk,nk);
option = optimoptions('fsolve','Display','none','FunctionTolerance',1e-7);

% Generate grids for Y and initial guess for V tilde
for ik = 1 : nk
    grid.Y(ik,:,:) = (grid.capital(ik).^aalpha.values'.* ...
        lss.^(1-aalpha.values'))*exp(z.values) + (1-ddelta)*grid.capital(ik);
    grid.Vtilde(ik,:,:) = ones(naalpha,nz) * bbeta*(uss)^(1-ppsi)*slope(ik);
end

iter_EGM1 = 1;
disp('Step 1: Endogenous Grid Method with fixed labor')
tic;
% Main loop
 while diff > tol
    
    for iz = 1 : nz
        for iaalpha = 1 : naalpha
            for ikp = 1 : nk
                % Calculate the derivative of Vtilde at each point of K on the K_t+1 grid
                derivativeVtilde = calculate_derivative(grid.Vtilde,grid.capital,ikp);
                
                % Solve for optimal c
                solve_c = @(c) 1-((1-bbeta)*(1-ppsi)*(log(c)-eeta*lss^2/2)^(-ppsi))*derivativeVtilde^(-1)/c;
                cStar = fsolve(solve_c,css,option);
                
                % Generate endogenous grid for Y and update Vend
                grid.Yend(ikp,iaalpha,iz) = cStar + grid.capital(ikp);
                grid.Vend(ikp,iaalpha,iz) = ((1-bbeta)*(log(cStar)-eeta*lss^2/2)^(1-ppsi)+grid.Vtilde(ikp,iaalpha,iz))^(1/(1-ppsi));           
            end
            
            % Interpolate Vend on grid.Y
            for iy = 1 : nk
                output = grid.Y(iy,iaalpha,iz);
                grid.Vupdated(iy,iaalpha,iz) = interpolate(grid.Vend(:,iaalpha,iz),output,grid.Yend(:,iaalpha,iz));                 
            end
            
        end
    end
    
    % Update Vtilde
    vecVupdated = reshape(grid.Vupdated,nk,naalpha*nz);
    vecVtildeUpdated = bbeta*(vecVupdated.^(1-ggamma)*mTransition').^((1-ppsi)/(1-ggamma));
    grid.VtildeUpdated = reshape(vecVtildeUpdated,nk,naalpha,nz);
    
    diff = abs(grid.Vtilde-grid.VtildeUpdated);
    diff = max(diff(:));
    grid.Vtilde = grid.VtildeUpdated;
    
    if (mod(iter_EGM1,10)==0 || iter_EGM1 ==1)
        fprintf(' Iteration = %d, Sup Diff = %2.8f\n', iter_EGM1, diff); 
    end
    iter_EGM1 = iter_EGM1 + 1;    
end
time_EGM1 = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 2: VFI with grid search to recover policy functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

policy.kVFI1 = zeros(nk,naalpha,nz);
policy.lVFI1 = zeros(nk,naalpha,nz);
grid.VFI1 = grid.Vupdated;
diff = inf;

iter_VFI1 = 1;
disp('Step 2: First Value Function Iteration to recover policy functions')
tic;
while iter_VFI1 == 1
    
    vecVFI = reshape(grid.VFI1.^(1-ggamma),nk,naalpha*nz);
    Vexpected = vecVFI * mTransition';
    Vexpected = reshape(Vexpected,nk,naalpha,nz);
    
    mVnew = zeros(nk,naalpha,nz);
    tempV = zeros(nz*naalpha*nk,1);
    tempK = zeros(nz*naalpha*nk,1);
    tempL = zeros(nz*naalpha*nk,1);

    for iz = 1 : nz
        for iaalpha = 1 : naalpha
            
            ikpStart = 1;
            for ik = 1 : nk

                capital = grid.capital(ik)
                tfp = z.values(iz);
                capitalElasticity = aalpha.values(iaalpha);

                VV = -1e5;
  
                for ikp = ikpStart : nk
                    capitalNext = grid.capital(ikp);
                    investment = capitalNext - (1-ddelta)*capital;

                    labor_choice = @(l) eeta*l*(exp(tfp)*capital^capitalElasticity*l^(1-capitalElasticity)+ ...
                    (1-ddelta)*capital-capitalNext)-(1-capitalElasticity)*exp(tfp)* ...
                    capital^capitalElasticity*l^(-capitalElasticity);

                    labor = fsolve(labor_choice,lss,option);
                    production = exp(tfp)*capital^capitalElasticity*labor^(1-capitalElasticity);
                    consumption = production - investment;


                    if consumption <= 0
                        break
                    end

                    const = log(consumption) - eeta*labor^2/2;

                    if const < 0
                        break
                    end

                    utility = ((1-bbeta)*const^(1-ppsi) + bbeta * Vexpected(ikp,iaalpha,iz)^ ...
                        ((1-ppsi)/(1-ggamma)))^(1/(1-ppsi));

                    if utility > VV
                        VV = utility;
                        optimalLabor = labor;
                        optimalCapital = capitalNext;
                        ikpStart = ikp;
                    else
                        break
                    end

                end

                mVnew(ik,iaalpha,iz) = VV;
                policy.kVFI1(ik,iaalpha,iz) = optimalCapital;
                policy.lVFI1(ik,iaalpha,iz) = optimalLabor;
            end
            
        end
    end
    
    diff = abs(mVnew-grid.VFI1);
    diff = max(diff(:));
    grid.VFI1 = mVnew;

    if (mod(iter_VFI1,10)==0 || iter_VFI1 ==1)
        fprintf(' Iteration = %d, Sup Diff = %2.8f\n', iter_VFI1, diff); 
    end
    iter_VFI1 = iter_VFI1 + 1;     
end
time_VFI1 = toc;

% Find kend and lend
grid.kend = zeros(nk,naalpha,nz);
grid.lend = zeros(nk,naalpha,nz);
for iz = 1 : nz
    for iaalpha = 1 : naalpha
        for ik = 1 : nk
            grid.kend(ik,iaalpha,iz) = interpolate(grid.capital,grid.capital(ik),policy.kVFI1(:,iaalpha,iz));
            grid.lend(ik,iaalpha,iz) = interpolate(policy.lVFI1(:,iaalpha,iz),grid.kend(ik,iaalpha,iz),grid.capital);
        end 
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 3: Apply EGM using the policy functions from the previous step
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

diff = inf;
vecVFI = reshape(grid.VFI1.^(1-ggamma),nk,naalpha*nz);
grid.Vtilde2 = bbeta*(vecVFI*mTransition').^((1-ppsi)/(1-ggamma));
grid.Vtilde2 = reshape(grid.Vtilde2,nk,naalpha,nz);

iter_EGM2 = 1;
disp('Step 3: Second EGM using the policy functions from Step 2')
tic;
while diff > tol
    
    for iz = 1 : nz
        for iaalpha = 1 : naalpha
            for ikp = 1 : nk
                % Calculate the derivative of Vtilde at each point of K on the K_t+1 grid
                derivativeVtilde = calculate_derivative(grid.Vtilde2,grid.capital,ikp);               
                
                % Solve for optimal c
                solve_c = @(c) 1-((1-bbeta)*(1-ppsi)*(log(c)-eeta*grid.lend(ikp,iaalpha,iz)^2/2)^(-ppsi))* ...
                    derivativeVtilde^(-1)/c;
                cStar = fsolve(solve_c,css,option);
                
                % Solve for kend using budget constraint
                solve_k = @(k) cStar + grid.capital(ikp) - exp(z.values(iz))*k^aalpha.values(iaalpha)* ...
                    grid.lend(ikp,iaalpha,iz)^(1-aalpha.values(iaalpha)) - (1-ddelta)*k;
                kend = fsolve(solve_k,kss,option);
                grid.kend2(ikp,iaalpha,iz) = kend;
                
                % Update value function
                grid.Vend2(ikp,iaalpha,iz) = ((1-bbeta)*(log(cStar)-eeta*grid.lend(ikp,iaalpha,iz)^2/2)^ ...
                    (1-ppsi)+grid.Vtilde2(ikp,iaalpha,iz))^(1/(1-ppsi));
            end
            
            % Interpolate Vend on the original grid
            for ik = 1 : nk
                grid.Vupdated2(ik,iaalpha,iz) = interpolate(grid.Vend2(:,iaalpha,iz),grid.capital(ik),grid.kend2(:,iaalpha,iz));  
                grid.lend(ik,iaalpha,iz) = interpolate(policy.lVFI1(:,iaalpha,iz),grid.kend2(ik,iaalpha,iz),grid.capital);
            end
                      
        end                       
    end

    
    % Update V tilde
    vecVupdated2 = reshape(grid.Vupdated2,nk,naalpha*nz);
    vecVtildeUpdated2 = bbeta*(vecVupdated2.^(1-ggamma)*mTransition').^((1-ppsi)/(1-ggamma));
    grid.VtildeUpdated2 = reshape(vecVtildeUpdated2,nk,naalpha,nz);
    
    diff = abs(grid.Vtilde2-grid.VtildeUpdated2);
    diff = max(diff(:));
    grid.Vtilde2 = grid.VtildeUpdated2;
    
    if (mod(iter_EGM2,10)==0 || iter_EGM2 ==1)
        fprintf(' Iteration = %d, Sup Diff = %2.8f\n', iter_EGM2, diff); 
    end
    iter_EGM2 = iter_EGM2 + 1;    
end
time_EGM2 = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 4: Second Value Function Iteration to generate policy functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

policy.kVFI2 = zeros(nk,naalpha,nz);
policy.cVFI2 = zeros(nk,naalpha,nz);
policy.lVFI2 = zeros(nk,naalpha,nz);
grid.VFI2 = grid.Vupdated2;
diff = inf;

iter_VFI2 = 1;
disp('Step 4: Second Value Function Iteration to generate policy functions')
tic;
while diff > tol
    
%     vecVFI = reshape(grid.VFI2.^(1-ggamma),nk,naalpha*nz);
%     Vexpected = vecVFI * mTransition';
%     Vexpected = reshape(Vexpected,nk,naalpha,nz);
%     
%     mVnew = zeros(nk,naalpha,nz);
%     tempV = zeros(nz*naalpha*nk,1);
%     tempK = zeros(nz*naalpha*nk,1);
%     tempL = zeros(nz*naalpha*nk,1);
% 
%     for iz = 1 : nz
%         for iaalpha = 1 : naalpha
%             
%             ikpStart = 1;
%             for ik = 1 : nk
% 
%                 capital = grid.capital(ik);
%                 tfp = z.values(iz);
%                 capitalElasticity = aalpha.values(iaalpha);
% 
%                 VV = -1e5;
%   
%                 for ikp = ikpStart : nk
%                     capitalNext = grid.capital(ikp);
%                     investment = capitalNext - (1-ddelta)*capital;
% 
%                     labor_choice = @(l) eeta*l*(exp(tfp)*capital^capitalElasticity*l^(1-capitalElasticity)+ ...
%                     (1-ddelta)*capital-capitalNext)-(1-capitalElasticity)*exp(tfp)* ...
%                     capital^capitalElasticity*l^(-capitalElasticity);
% 
%                     labor = fsolve(labor_choice,lss,option);
%                     production = exp(tfp)*capital^capitalElasticity*labor^(1-capitalElasticity);
%                     consumption = production - investment;
% 
% 
%                     if consumption <= 0
%                         break
%                     end
% 
%                     const = log(consumption) - eeta*labor^2/2;
% 
%                     if const < 0
%                         break
%                     end
% 
%                     utility = ((1-bbeta)*const^(1-ppsi) + bbeta * Vexpected(ikp,iaalpha,iz)^ ...
%                         ((1-ppsi)/(1-ggamma)))^(1/(1-ppsi));
% 
%                     if utility > VV
%                         VV = utility;
%                         optimalLabor = labor;
%                         optimalCapital = capitalNext;
%                         ikpStart = ikp;                 
%                     else
%                         break
%                     end
% 
%                 end
% 
%                 mVnew(ik,iaalpha,iz) = VV;
%                 policy.kVFI2(ik,iaalpha,iz) = optimalCapital;
%                 policy.lVFI2(ik,iaalpha,iz) = optimalLabor;
%                 policy.cVFI2(ik,iaalpha,iz) = consumption;
%             end
%         end
%     end
%     
%     diff = abs(mVnew-grid.VFI2);
%     diff = max(diff(:));
%     grid.VFI2 = mVnew;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    val = @(k) value_function(k,grid.VFI2,option);
    tempV = zeros(nk,naalpha,nz);

    for ik = 1 : nk
        capital = grid.capital(ik);
        for iaalpha = 1 : naalpha
            capitalElasticity = aalpha.values(iaalpha);
            for iz = 1 : nz
                tfp = z.values(iz);

                capitalNext = fminbnd(val,kmin,kmax);
                policy.kVFI2(ik,iaalpha,iz) = capitalNext;
                policy.iVFI2(ik,iaalpha,iz) = investment;
                policy.cVFI2(ik,iaalpha,iz) = consumption;
                policy.lVFI2(ik,iaalpha,iz) = labor;

                tempV(ik,iaalpha,iz) = -val(capitalNext);
            end
        end
    end
    diff = abs(tempV-grid.VFI2);
    diff = max(diff(:));  
    grid.VFI2 = tempV;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if (mod(iter_VFI2,10)==0 || iter_VFI2 ==1)
        fprintf(' Iteration = %d, Sup Diff = %2.8f\n', iter_VFI2, diff); 
    end
    iter_VFI2 = iter_VFI2 + 1;     
end
time_VFI2 = toc;

%% Plot the functions
v_fun = reshape(grid.VFI2,nk,naalpha*nz);
g_k = reshape(policy.kVFI2,nk,naalpha*nz);
g_c = reshape(policy.cVFI2,nk,naalpha*nz);
g_l = reshape(policy.lVFI2,nk,naalpha*nz);

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
    ylim(ylims{i}); xlim([180 360]);
    
    subplot(2,1,2)
    plot_function = ['plot(grid.capital,' series_names{i} '(:,7:9))'];
    evalin('base',plot_function);
    title([title_names{i} ', z=0'])
    legend('\alpha = 0.25','\alpha = 0.3','\alpha = 0.35');
    xlabel('Capital Today','FontSize',7)
    ylabel(ylabel_names{i},'FontSize',7)
    ylim(ylims{i}); xlim([180 360]);
    saveas(gcf,['..' filesep 'Output' filesep 'Q4_EGM_' file_label{i} '.png']);
    close(gcf)
end

%% Plot impulse response functions
shock_variable = {'z','alpha'};

T = 60;
cPath = [policy.cVFI2(95,2,3) zeros(1,T-1)];
cPath_percentdev = zeros(1,T);
kPath = [policy.kVFI2(95,2,3), zeros(1,T-1)];
kPath_percentdev = zeros(1,T);
lPath = [policy.lVFI2(95,2,3) zeros(1,T-1)];
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
        cPath(t) = interpolate(policy.cVFI2(:,aPath(t),zPath(t)),kPath(t-1),grid.capital);
        cPath_percentdev(t) = (log(cPath(t))-log(policy.cVFI2(95,2,3)))*100;

        kPath(t) = interpolate(policy.kVFI2(:,aPath(t),zPath(t)),kPath(t-1),grid.capital);
        kPath_percentdev(t) = (log(kPath(t))-log(policy.kVFI2(95,2,3)))*100;

        lPath(t) = interpolate(policy.lVFI2(:,aPath(t),zPath(t)),kPath(t-1),grid.capital);
        lPath_percentdev(t) = (log(lPath(t))-log(policy.lVFI2(95,2,3)))*100;
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
    
    saveas(gcf,['..' filesep 'Output' filesep 'Q4_EGM_IRF_' shock_variable{i} '.png']);
    close(gcf)
end

%% Compute Euler Equation Errors
[euler_error_EGM,max_error_EGM] = euler_error_VFI(aalpha,bbeta,ggamma,ddelta,ppsi, ...
    eeta,grid.VFI2,policy.cVFI2,policy.lVFI2,policy.kVFI2,z,grid.capital,naalpha,nz,nk);

%% Save results
save(['..' filesep 'Output' filesep 'Q4_EGM.mat'])
