%% MASTER: Validation of mmPLRNN on experimental data (PLRNN vs. mmPLRNN):
path(path,pwd);
config.pat= pwd; 
config.type = 'fMRI/'; %specify data folder/typ used for the training
config.sett = 'NBKO_PLRNN_dataset_RestReference.mat'; %Name of preprocessed dataset
config.noinp = 1; %Set according to paper experiments, i.e. training without recognition of external inputs 
config.Full = 1; %Set according to paper experiments, i.e. non-trial-wise training
config.numlat = 20; %Set according to paper experiments, i.e. 20 latent states
config.patnum = 1; %Define accordin

[num_m, FULL, NoInp] = Training_Unimodal_vs_MultimodalOriginal(config);

config.num_smp = 7;
config.nahead = 10:2:14;

Evaluation_NetworkComparison_PowerSpec_MSE(config)