%% MASTER: Validation of mmPLRNN on noisy Lorenz (PLRNN vs. mmPLRNN):
addpath(genpath(pwd));
config.snr=2; %Set according to the number of trajectories generated, i.e number of networks to be trained
config.sysnr=1:config.snr;
config.pat = [pwd,'/LorenzData/Data/3Dclassification/NoisyLorenz/'];
config.patnet = [pwd, '/LorenzData/EM/3Dclassification/NoisyLorenz/'];
config.tlen = 1000;
config.s2 =0.001;
config.sobs = 0.1;


[pat2, run] = Evaluation_of_LorenzNoise(config);

config.patR = [pwd,'/LorenzData/Results/NoisyLorenz/'];

NoisyLorenz_Analysis_PaperThesis(config)
