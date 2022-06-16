%% Network Training on experimental data (PLRNN vs. mmPLRNN):
function [M, FULL, NoInp] = Training_Unimodal_vs_MultimodalOriginal(config)
% Network can be trained on trial-based data as well as full time series (set FULL parameter accordingly)
% Network can be trained using external inputs
%% Step 1: Data formatting and preprocessing
clc

pat = config.pat
path(path,pat);

% Preprocessing (How to bring data into according format):
    %dataset should have following format:
        %File should only contain dataset = cell with 1 x #Patients (collection of any data type)
        %data = dataset{1,pt}- struct containing data for each indv. pat.
        %data.X - Gaussian data comes as cell array with 1 x #trial entries
            %data.X{1,trial} - array N x T_trail
        %data.C_ - behavioral data (categories) same format as X cell-wise
            %data.C_{1,trial} - array Kk x T_trial
        %data.Inp - external Inputs same format as gaussian data
            %data.Inp{1,trial} - array K x T_trial

% Load Data from according path:
type = config.type;
sett = config.sett;
% type = 'fMRI/'; %specify data folder/typ used for the training
% sett = 'NBKO_PLRNN_dataset.mat'; %Name of preprocessed dataset
fnOut{1}=[pat '/Data/Training/' type sett];
    
d = load(fnOut{1});
dataset=d.Data;

if strcmp(config.patnum,'full')
    patnum = length(dataset);
else
    patnum = config.patnum;
end


% Set Folder for Trained Networks of according Dataset:
fnL{1}=[pat '/Data/Evaluation/NetComparison/' type];
mkdir(fnL{1});

%% Step 2: Training Loop over Patients

%Set training conditions:
FULL=config.Full; 
NoInp=config.noinp; 

%Set global hyperparameters (only first one should be set freely rest should stay more or less untouched, i.e. NR unstable):
M= config.numlat;
lam=0; %ALWAYS 0
NR.g=0.0005; %Learning rate of categorical model update via NR (M-step)
NR.bound=0.01; %Convergence bound of categorical NR (M-step)
NR.use=true; %Change to False if PLRNN training with single modality
NR.fixedBeta=0; %Fixes update of categorical model parameters (M-step)

% Set Sub-Folders for Trained Networks according to training conditions:
fnL{1}=[pat '/Data/Evaluation/NetComparison/' type];
if FULL
    fnL{2}=[fnL{1} '/FullTS'];
    mkdir(fnL{2})
    if NoInp
        fnL{3}=[fnL{2} '/NoInp'];
    else
        fnL{3}=[fnL{2} '/Inp'];
    end
else
    fnL{2}=[fnL{1} '/TrialWise'];
    mkdir(fnL{2})
    if NoInp
        fnL{3}=[fnL{2} '/NoInp'];
    else
        fnL{3}=[fnL{2} '/Inp'];
    end
end

mkdir(fnL{3});
fnL{4} = [fnL{3} '/m' num2str(M) '/'];
mkdir(fnL{4});

for pt=1:patnum
%CHOOSE patient (data set which we train on)
    data=dataset{1,pt};

   if FULL==1
       data.Xn=data.X;
      [data]=make_fTS(data);
   else
       data.Xn=data.X;
   end
   
%Local hyperparameters:
   % Define CtrPar/NR struct:
   CtrPar.tol = 1*10^(-2);
   CtrPar.MaxIter = 51; %Max iterations EM algo
   CtrPar.tol2 = 0;
   CtrPar.eps = 1*10^(-5);
   CtrPar.flipOnIt = 10;
   CtrPar.exp=false;
   CtrPar.boundE=0.01; %Convergence bound Semi-NR (i.e. E-step)
   CtrPar.gE=1; %Learning rate for the Semi-NR (do not change)
   CtrPar.maxItr=6; % Maximum Repitions of the Semi-NR Procedure (only increase if necessary)
   CtrPar.outbnd=10;% Outerbound Estep
   CtrPar.FinOpt = 0;   % quad. prog. step at end of E-iterations
   CtrPar.fixedS = false;   % S to be considered fixed or to be estimated
   CtrPar.fixedC = false;   % C to be considered fixed or to be estimated
   CtrPar.fixedB = false;   % B to be considered fixed or to be estimated
   CtrPar.fixedG= false;
   
   K=size(data.Inp{1},1);
   Kk=size(data.C_{1},1);
   T=size(data.Xn{1},2);
   
   if FULL==1
       numTr=1;
       Tsum=numTr*T; 
   else
       numTr=length(data.Xn);
       Tsum=numTr*T;
   end
    
   lam_=lam/Tsum(end);
   Cstr.lam=lam_;
   Cstr.Lb=[]; Cstr.Ub=[]; Cstr.XZspl=[]; 

   for sr=1:5
        rng(sr,'twister')
        m=M;
        n=size(data.X{1},1);

        InPar=struct;
        a=-1; b=1;
        InPar.A=diag(diag(a+(b-a).*rand(m)));
        InPar.W= a+(b-a).*rand(m);
        InPar.W=InPar.W.*[ones(m,m)-eye(m)];

        stationaritycond=max(abs(eig(InPar.A+InPar.W)));
        reduc=0.95;    
        while stationaritycond>1 || stationaritycond==1
            InPar.W = InPar.W*reduc;
            InPar.A = InPar.A*reduc;
            stationaritycond=max(abs(eig(InPar.A+InPar.W)));
        end
        s=.001;
        InPar.C=a+(b-a).*rand(m,K);
        InPar.S=diag(diag(s*rand(m)));
        if FULL==1
            if NoInp
                Inp{1}=zeros(K,T);
                InPar.Inp=Inp;
                clear Inp
            else
                InPar.Inp=data.Inp;
            end
        else
            if NoInp %Inputs to 0 if no inputs used
                Inp__=cell(1,length(data.Inp));
                for nbs=1:length(data.Inp)
                    Inp__{1,nbs}=zeros(K,T);
                end
                InPar.Inp=Inp__;
            else
                InPar.Inp=data.Inp;
            end
        end
        InPar.mu0=rand(m,1);
        InPar.B=a+(b-a).*rand(n,m);
        InPar.G=diag(diag(s*rand(n)));
        InPar.h=a+(b-a).*rand(m,1);
        InPar.Beta=a+(b-a).*rand(m,Kk-1);
        InPar.X=data.Xn;
        InPar.C_=data.C_;
        Cstr.ChunkList=[1 numTr+1];

        %Regularization Parameters (REG NOT USED):
        opt=1; 
        NR.mod=0;
        tau=10^6/Tsum;
        
    try
%% Multimodal Training (mmPLRNN):
    NR.use=true; %Change to False if only Gaussian timeseries is used for training
    CtrPar.tol = 1*10^(-2);
    CtrPar.LAR=0;
    if FULL==1
        Cstr.ChunkList=[1 2];
    else
        Cstr.ChunkList=[1 length(InPar.X)+1];
    end
    
    if NoInp
        [net,OutPar]=runPLRNN_initproc_20_MC_NS_NI(InPar,CtrPar,Cstr,NR,m,opt,tau); %Optimized for experimental data, NOTE: Regularization NOT used here
    else     
        [net,OutPar]=runPLRNN_initproc_20_MC_NS(InPar,CtrPar,Cstr,NR,m,opt,tau); %Optimized for experimental data, NOTE: Regularization NOT used here
    end

    %Unpack parameter estimates:
    mu0=OutPar.mu0;
    B=OutPar.B;
    G=OutPar.G;
    A=OutPar.A;
    W=OutPar.W;
    C=OutPar.C;
    h=OutPar.h;
    S=OutPar.S;
    Beta=OutPar.Beta;
    LL=OutPar.LL;
    Vest=OutPar.Vest;
    Xn=OutPar.X;
    lam=Cstr.lam;
    Ezi=OutPar.Ezi;
    Lb=Cstr.Lb;
    Ub=Cstr.Ub;
    M=m;
    C_=InPar.C_;
    Inp=InPar.Inp;
    


    fnOut{4}=[fnL{4} '/mmPLRNN_m' num2str(m) 'pat_' num2str(pt) '_init_' num2str(sr) '.mat'];
    save(fnOut{4},'mu0','B','G','W','A','S','C','C_','h','Beta','Ezi', ...
            'LL','Xn','Inp','M','Vest','lam','Lb','Ub');     


    
%% Unimodal Training (PLRNN):
    NR.use=false; %Drops calculation of categorical model in M-step and drops term for C-model in E-step
    CtrPar.tol = 1*10^(-2);
    CtrPar.LAR=0; % Set to control use with or without regularization!
    Cstr.ChunkList=[1 length(InPar.X)+1];
    
    if NoInp
        [net,OutParG]=runPLRNN_initproc_20_MC_NS_NI(InPar,CtrPar,Cstr,NR,m,opt,tau);
    else
        [net,OutParG]=runPLRNN_initproc_20_MC_NS(InPar,CtrPar,Cstr,NR,m,opt,tau);
    end


    %Unpack parameter estimates:
    mu0G=OutParG.mu0;
    BG=OutParG.B;
    GG=OutParG.G;
    AG=OutParG.A;
    WG=OutParG.W;
    CG=OutParG.C;
    hG=OutParG.h;
    SG=OutParG.S;
    LLG=OutParG.LL;
    VestG=OutParG.Vest;
    Xn=OutParG.X;
    lam=Cstr.lam;
    EziG=OutParG.Ezi;
    Lb=Cstr.Lb;
    Ub=Cstr.Ub;
    M=m;
    C_=InPar.C_;
    Inp=InPar.Inp;
    

    fnOut{5}=[fnL{4} '/Sparse_PLRNN_m' num2str(m) 'pat_' num2str(pt) '_init_' num2str(sr) '.mat'];
    save(fnOut{5},'mu0G','BG','GG','WG','AG','SG','CG','C_','hG','EziG', ...
            'LLG','Xn','Inp','M','VestG','lam','Lb','Ub');

        
    catch
        disp('Training not converged')
    end
    end
 end    
end
