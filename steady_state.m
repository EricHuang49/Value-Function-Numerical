function [css,kss,lss,eeta] = steady_state(aalpha,bbeta,ddelta,z,production)
    kl = ((1-bbeta*(1-ddelta))/(bbeta*aalpha*exp(z)))^(1/(aalpha-1));
    lss = production/(exp(z)*kl^aalpha);
    css = (exp(z)*kl^aalpha-ddelta*kl)*lss;
    kss = lss * kl;
    eeta = (1-aalpha)*exp(z)*kl^aalpha/(lss*css);
end