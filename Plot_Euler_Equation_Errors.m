clear all;clc;

load(['..' filesep 'Output' filesep 'Q3_VFI.mat'],'grid','max_error_VFI')
load(['..' filesep 'Output' filesep 'Q4_EGM.mat'],'max_error_EGM')
load(['..' filesep 'Output' filesep 'Q5_Accelerator.mat'],'max_error_Accelerator')
plot(grid.capital,[max_error_VFI' max_error_EGM' max_error_Accelerator'])
hold on
load(['..' filesep 'Output' filesep 'Q6_Multigrid.mat'],'grid','max_error_Multigrid')
plot(grid.capital,max_error_Multigrid')
xmin = min(grid.capital);
xmax = max(grid.capital);
load(['..' filesep 'Output' filesep 'Q7_Chebyshev.mat'])
plot(grid.k_complete,max_error_Chebyshev)
legend('VFI','EGM','Accelerator','Multigrid','Chebyshev')
xlim([xmin xmax])

saveas(gcf,['..' filesep 'Output' filesep 'EEE_all.png'])