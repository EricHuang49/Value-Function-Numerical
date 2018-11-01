clear;clc;

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
production          = 100;

nz                  = length(z.values);
naalpha             = length(aalpha.values);
nk                  = 100;

%% Calculate steady state
[css,kss,lss,eeta] = steady_state(aalpha.values(2),bbeta,ddelta,z.values(3),production);

%% Generate grids
kmin = (1-0.3) * kss;
kmax = (1+0.3) * kss;
grid.k = linspace(kmin,kmax,nk);

%% Value function iteration
V = zeros(nz,naalpha,nk);
diff = inf;

while diff > tol
    Vexpected = V;
    for iaalpha = 1 : naalpha
        for iz = 1 : nz
            Vp = reshape(Vexpected,[],nz);
            Vp = Vp * z.transition(iz,:)';
            Vp = reshape(Vp,[],naalpha);
            Vp = Vp * aalpha.transition(iaalpha,:)';
            Vexpected(iz,iaalpha,:) = reshape(Vp,1,nk);           
        end
    end
    
    Vcurr = zeros(nz,naalpha,nk);
    tempV = zeros(nz*naalpha*nk,1);
    for ind = 1 : nz*naalpha*nk
        ik = floor(mod(ind-0.05,nk))+1;
        iz = mod(floor((ind-0.05)/nk),nz)+1;
        iaalpha = mod(floor((ind-0.05)/(nk*nz)),naalpha)+1;
        
        capital = grid.k(ik);
        tfp = z.values(iz);
        capitalElasticity = aalpha.values(iaalpha);
        
        labor = (production / (exp(tfp)*capital^capitalElasticity))^(1/(1-capitalElasticity));
        VV = -1e5;
        
        for ikp = 1 : nk
            capitalNext = grid.k(ikp);
            investment = capitalNext - (1-ddelta)*capital;
            consumption = production - investment;
            
            if consumption <= 0
                break
            end
            
            const = log(consumption) - eeta*labor^2/2;
            
            if const < 0
                break
            end
            
            utility = const^(1-ppsi) + bbeta * Vexpected(iz,iaalpha,ikp);
            
            if utility > VV
                VV = utility;
            end
            
        end
        
        tempV(ind) = VV;
    end
    
    for ind = 1 : nz*naalpha*nk
        ik = floor(mod(ind-0.05,nk))+1;
        iz = mod(floor((ind-0.05)/nk),nz)+1;
        iaalpha = mod(floor((ind-0.05)/(nk*nz)),naalpha)+1;
        
        Vcurr(iz,iaalpha,ik) = tempV(ind);
    end
    
    diff = abs(Vcurr-V);
    diff = max(diff(:));
    V = Vcurr;
end