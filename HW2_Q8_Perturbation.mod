//----------------------------------------------------------------
// 0. Housekeeping
//----------------------------------------------------------------

close all

//----------------------------------------------------------------
// 1. Endogenous variables (12=5+2+1+1+1+1+1)
//----------------------------------------------------------------

var 

// Allocation (5)
y c i l k 

// Prices (2)
r w

// Productivity (1)
z

// Elasticity of substitution parameter (1)
aalpha

// Stochastic discount factor (1)
m

// Value function (1)
V

// New variable for expected value (1)
xi;

//----------------------------------------------------------------
// 2. Exogenous variables (2)
//----------------------------------------------------------------
 
varexo 

ez eaalpha;

//----------------------------------------------------------------
// 3. Parameters
//----------------------------------------------------------------

parameters 

// Preferences
bbeta ppsi ggamma eeta

// Technology
ddelta rrhoz rrhoaalpha

// S.D.'s stochastic processes
ssigmaz ssigmaaalpha;

//----------------------------------------------------------------
// 4. Calibration
//----------------------------------------------------------------

// Preferences
bbeta      = 0.99;
ppsi       = 0.5;
ggamma     = 10;

// Technology
ddelta     = 0.1;
rrhoz      = 0.95;  
rrhoaalpha = 0.9;

// Stochastic processes
ssigmaz      = 0.005;
ssigmaaalpha = 0.01;

kl_ss = ((1-bbeta*(1-ddelta))/(bbeta*0.3))^(0.3-1);
l_ss  = 100 / (kl_ss^0.3);
r_ss  = 1/bbeta-1+ddelta;
k_ss  = l_ss * kl_ss;
y_ss  = (k_ss^0.3)*(l_ss^(1-0.3));
w_ss  = (1-0.3)*y_ss/l_ss;
i_ss  = ddelta*k_ss;
c_ss  = y_ss-i_ss;
eeta  = (1-0.3)*kl_ss^0.3/(l_ss*c_ss);
u_ss  = log(c_ss) - eeta*l_ss^2/2;

//----------------------------------------------------------------
// 5. Model
//----------------------------------------------------------------

model; 
  
  // 1. Euler equation return on private capital
  1 = m(+1)* (1+r(+1)-ddelta);
  
  // 2. Rental rate of capital
  r = aalpha*y/k(-1);
  
  // 3. Static condition leisure-consumption
  eeta*l*c = w;

  // 4. Wages
  w = (1-aalpha)*y/l;

  // 5. Law of motion for private capital
  k = (1-ddelta)*k(-1)+i;

  // 6. Production function
  y = (k(-1)^aalpha)*((exp(z)*l)^(1-aalpha));

  // 7. Resource constraint of the economy 
  c+i = y;

  // 8. Productivity process 
  z = rrhoz*z(-1)+ssigmaz*ez;

  // 9. Elasticity parameter process
  aalpha = 0.03 + rrhoaalpha*aalpha(-1)+ssigmaaalpha*eaalpha;

  // 10. Stochastic discount factor
  m = bbeta*((log(c(-1))-eeta*l(-1)^2/2)^(ppsi)*c(-1))/((log(c)-eeta*l^2/2)^(ppsi)*c)* (V/xi(-1))^(ppsi-ggamma);

  // 11. Value function
  V = ((1-bbeta)*(log(c)-eeta*l^(2)/2)^(1-ppsi) + bbeta*xi^(1-ppsi))^(1/(1-ppsi));

  // 12. New variable for expected value
  xi^(1-ggamma) = V(+1)^(1-ggamma);

end;

//----------------------------------------------------------------
// 6. Computation
//----------------------------------------------------------------

initval;
  y = y_ss;
  c = c_ss;
  i = i_ss;
  l = l_ss;
  k = k_ss;
  r = r_ss;
  w = w_ss;
  z = 0;
  aalpha = 0.3;
  m = bbeta;
  V = u_ss;
  xi = u_ss;
  ez = 0;
  eaalpha = 0;
end;

shocks;
  var ez = 1;
  var eaalpha = 1;
end;

steady;

check;

stoch_simul(hp_filter = 1600, irf = 50, order = 3) V c k l;