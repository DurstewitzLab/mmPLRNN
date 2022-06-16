%% MASTER: Validation of mmPLRNN on reduced Lorenz (PLRNN vs. mmPLRNN):
addpath(genpath(pwd));
snr=2; %Set according to number of trajectories/number of networks to train
sysnr=1:snr;
pat = [pwd,'/LorenzData/Data/3Dclassification/RedLorenz/Yclass/'];
pat2 = [pwd,'/LorenzData/EM/3Dclassification/RedLorenz/Yclass/'];


[pat2, run3]=Evaluation_of_LorenzDimRed(sysnr,pat,pat2);

patR = [pwd,'/LorenzData/Results/ReducedLorenz/'];

Analysis_redLorenz_system(pat2,pat,patR,snr)
