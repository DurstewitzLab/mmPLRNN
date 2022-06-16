%% MASTER: Cross-modal inference ability validation of mmPLRNN on experimental data (PLRNN vs. mmPLRNN):
addpath(genpath(pwd));
config.pat= pwd; 
config.type = 'fMRI/'; %specify data folder/typ used for the training
config.sett = 'NBKO_PLRNN_dataset.mat'; %Name of preprocessed dataset for cross-modal inference 
config.noinp = 1; %Set according to paper experiments, i.e. training without recognition of external inputs 
config.numlat = 20; %Set according to paper experiments, i.e. 20 latent states
config.patnum = 1; %Define accordin

[num_m, NoInp] = CrossValidation_PLRNNonfMRIData(config);



Evaluation_CrossValidation_ForTrialData(config)
