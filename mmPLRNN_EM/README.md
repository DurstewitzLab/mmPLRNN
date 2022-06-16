# Multi-modal PLRNN: Reconstructing nonlinear dynamical systems from multi-modal timeseries

### Code & Data
[![MATLAB]  >=2020a]
Package contains fMRI data based on [Koppe et. al](https://pubmed.ncbi.nlm.nih.gov/25019681/) 

## This package contains the code base for the EM-based multi-modal PLRNN and all experiments according to publication
[Kramer et. al. 2021](https://arxiv.org/abs/2111.02922)



### 0. Prerequisits

- Matlab experience
- Matlab grafic interface 


### 1. Instructions

- Clone Repository
- Move to own Project directory
- add package folder and subfolders to matlab path 
- make sure to work in package directory as current path, e.g. .../CodeEM/


### 2. Data

all trained networks used to produce results as displayed in publication are provided in a sparse format for evaluation in PLRNN_VI Code Package


#### 2.1 Benchmark
* A. training data* 
- two examples trajectories for each lorenz experiments are included in folders '/LorenzData/'
*data simulation*
- adjustable: s1 - (latent) process noise, sobs - observation noise 
- script is provided 'Application/Utils_Lorenz/Simulation/Get_LorenzTraj_withClassLabel.m', with all hyperparameters according to publication experiments of the *Noisy Lorenz*
- *Incomplete Lorenz* set sobs = 0.001, dim = 1, d = 2 (lorenz trajectories without y dimension)


* B. trained networks* 
- mat files with parameters of trained network (MM - mmPLRNN, Gauss - PLRNN, Init - IntialValues) are NOT provided in this package for SIZE reasons and according results can be found in sparse format in PLRNN_VI package

[Koppe et. al](https://pubmed.ncbi.nlm.nih.gov/25019681/)
- '/Data/Training/NBKO_PLRNN_dataset_RestReference.mat' used for network comparison and in order to produce Fig.3(A,B) and Fig.4
- '/Data/Training/NBKO_PLRNN_dataset.mat' used for CrossValidation in the cross-modality prediction task and in order to produce Fig.3C

- files containing parameters of *trained networks* for each patient(pat) and different initializations (init) are in VAE package

### 3. Network
- all code for the PLRNN construction, EM procedure (see Appendix) and network related scripts are contained in '/Network/'



### 4. Experiments
- for all applications make sure to work in package directory as your current path i.e '.../CodeEM/'
- All experiments can be run by executing the according MATLAB code in 'Masters_'+ ExperimentName + 'Validation.m'-files in '/Application'
- *Disclaimer*: We maintain best settings for the EM training, hence only a few hyperparameters are adjustable in the experiment runs. Further Hyperparameters can be adjusted in the according training procedures:
'Training_Unimodal_vs_MultimodalOriginal.m'
'CrossValidation_PLRNNonfMRIData.m'
'Evaluation_of_LorenzNoise.m'
'Evaluation_of_LorenzDimRed.m'

- *Disclaimer*: if interested in analysis part only comment out training

#### 4.1. Benchmark
*A. Incomplete Lorenz*
- Script of experiment for the comparison between PLRNN and mmPLRNN '/Application/Master_redLorenz_Validation.m', script can be executed in current settings to reproduce paper results. 
- Results will be saved in '/LorenzData/Results/ReducedLorenz/'
- Parameters which can be adjusted are: config.snr - number of trajectories used in experiment (each trajectory trains indv. network) 

*B. Noisy Lorenz*
- Script of experiment for the comparison between PLRNN and mmPLRNN '/Application/Master_NoisyLorenz_Validation.m', script can be run in current settings to reproduce paper results. 
- Results will be saved in '/LorenzData/Results/NoisyLorenz/'
- Parameters which can be adjusted are: config.snr - number of trajectories used in experiment (each trajectory trains indv. network)  


#### 4.2. Emperical Example: fMRI recordings
[Koppe et. al](https://pubmed.ncbi.nlm.nih.gov/25019681/)

*A. General*
- config.NoInp - Network is trained without recognition of external inputs can be set to 1 if inputs are part of fitting procedure
- config.type - sets the folder for the data used for training, if new data (data should be formatted according to preprocessing comment in 'Training_Unimodal_vs_MultimodalOriginal.m') is added to '/Data/Training/' change to according directory


*B. Network Comparison*
- Script of experiment for the comparison between PLRNN and mmPLRNN '/Application/Master_Experimental_Validation.m', script can be run in current settings to reproduce paper results. 
- Parameters which can be adjusted are: config.numlat - number of latent states, config.patnum - number of patients for training (each patient trains indv. network), config.Full - can be set to zero to train the network on batches, i.e faster training (can result in worsned prediction performance)

*C. Cross-modal prediction*
- Script of experiment for the cross-modality predictions '/Application/Master_CrossModalValidation.m', script can be run in current settings to reproduce paper results. 
- Parameters which can be adjusted in for further experiments are: config.numlat - number of latent states, config.patnum - number of patients for training (each patient trains indv. network)

### 5. Plot Routines

- All plots and evaluations are produced using python, see PLRNN_VI python package for this publication also contained in supplement [VAE package, Kramer et. al.]()
- Paths in VAE code have to be adjusted accordingly
- *Paper Results for Plotting* the benchmarking reults are also provided in VAE Package
- *Paper Results for Plotting* the results on the experimental data are also provided in VAE Package
    


