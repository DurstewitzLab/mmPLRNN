%% Evaluation of the Cross-validation-fMRI:
function Evaluation_CrossValidation_ForTrialData(config)
%% DISCLAIMER: This Version has to be adjusted according to your paths 
clc
close all
%% Step 1: Choose dataset according to training settings and initiation
pat = config.pat;
% Load Data from according path:
type = config.type; %specify data folder/typ used for the training
set = config.sett; %Name of preprocessed dataset
fnOut{1}=[pat '/Data/Training/' type set]; 
d = load(fnOut{1});
dataset=d.Data;

if strcmp(config.patnum,'full')
    patnum = length(dataset);
else 
    patnum = config.patnum;
end
    

% Set Folder for Trained Networks of according Dataset:
fnL{1}=[pat '/Data/Evaluation/CV/' type];
fnL{2}=[pat '/Data/Evaluated/CV/' type];
mkdir(fnL{2});

%% Step 2: CV Training Loop over Patients and all trials

%Set training conditions:
NoInp = config.noinp;
M=config.numlat;


%Set global hyperparameters (only first one should be set freely rest should stay more or less untouched, i.e. NR unstable):
ln=M;
lam_=0; %ALWAYS 0
NR.g=0.0005; %Learning rate of categorical model update via NR (M-step)
NR.bound=0.01; %Convergence bound of categorical NR (M-step)
NR.use=true; %Change to False if PLRNN training with single modality
NR.fixedBeta=0; %Fixes update of categorical model parameters (M-step)

% Set Sub-Folders for Trained Networks according to training conditions:
if NoInp
    fnL{3}=[fnL{2} '/NoInp'];
    fnL{5}=[fnL{1} '/NoInp'];
else
    fnL{3}=[fnL{2} '/Inp'];
    fnL{5}=[fnL{1} '/Inp'];
end


mkdir(fnL{3});
fnL{4} = [fnL{3} '/m' num2str(M) '/'];
fnL{6} = [fnL{5} '/m' num2str(M) '/'];
mkdir(fnL{4});

%% Left-out trial training:

for pt=1:patnum

j=0;


%% Calling the function:

% pt = patient_numbers(pn);
data=dataset{1,pt};
fMRI=dataset{pt};
nbatch=length(fMRI.X);
C_=fMRI.C_;

% Predict behavioral data
K=size(C_{1},1);
    
  for k=1:nbatch
    try
      m = ln;
      j=j+1;
        
      fnOut{3}=[fnL{6} 'data_sparse_PLRNN_m' num2str(m) 'pat_' num2str(pt) '_cv_' num2str(k) '.mat'];
      load(fnOut{3});
    
        NR.mod=0;

        T=size(X_{1,1},2);
        lam=lam_/(T*5);



        pertub=1e-3;
        InPar.S=OutPar.S;
        InPar.Inp=Inp{k};
        InPar.X=X_{k};
        InPar.C_=C_{k};
        Inp_ =Inp{k};
        InPar.mu0=rand(M,1);
        InPar.G=cell2mat(OutPar.G);
        InPar.B=cell2mat(OutPar.B);
        InPar.Beta=cell2mat(OutPar.Beta);
        InPar.W=OutPar.W;
        InPar.C=OutPar.C;
        InPar.A=OutPar.A;
        InPar.h=OutPar.h;

        InPar.X_=X_{k};
        CtrPar.tol = 1*10^(-2);
        CtrPar.LAR=0;
        ChunkList=[1 2];
        CtrPar.tol = 1*10^(-2);
        CtrPar.MaxIter = 51; %Max iterations EM algo
        CtrPar.tol2 = 0;
        CtrPar.eps = 1*10^(-5);
        CtrPar.flipOnIt = 10;
        CtrPar.exp=false;
        CtrPar.boundE=0.01;
        CtrPar.gE=1;
        CtrPar.maxItr=6;
        CtrPar.outbnd=10;% Outerbound Estep
        CtrPar.FinOpt = 0;   % quad. prog. step at end of E-iterations
        CtrPar.fixedS = false;   % S to be considered fixed or to be estimated
        CtrPar.fixedC = false;   % C to be considered fixed or to be estimated
        CtrPar.fixedB = false;   % B to be considered fixed or to be estimated
        CtrPar.fixedG= false;
        CtrPar.fixedW = false;   % B to be considered fixed or to be estimated
        CtrPar.fixedA= false;
        CtrPar.fixedh= false;
        CtrPar.fixedmu0= false;
        CtrPar.Cceil= 10;

        reg=0;

        Lb=[]; Ub=[]; XZspl=[]; 

        [OutParN,Ezi,Vest,Ephizi,Ephizij,Eziphizj,LL]=EMiterMultDS_regMC(CtrPar,InPar,NR,XZspl,Lb,Ub,lam,ChunkList,reg);

        fnOut{4}=[fnL{4} 'CV_sparse_PLRNN_m' num2str(m) 'pat_' num2str(pt) '_cv_' num2str(k) '.mat'];
        save(fnOut{4},'OutPar','OutParN','Ezi','X_','C_','Inp');

    catch
        disp('set not converged')

    end
  end
end
%% SAVE:
fnOut{5}=[fnL{4} 'CV_sparse_PLRNN_m' num2str(20) '_cv_Eval_all.mat'];
save(fnOut{5});
