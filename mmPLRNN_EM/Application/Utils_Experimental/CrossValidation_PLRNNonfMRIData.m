%% Cross-Validation Training:
function [M,NoInp]= CrossValidation_PLRNNonfMRIData(config)
%% Step 1: Data formatting and preprocessing
clc

pat = config.pat;
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
fnOut{1}=[pat '/Data/Training/' type sett];
    
d = load(fnOut{1});
dataset=d.Data;
d = load(fnOut{1});
dataset=d.Data;

if strcmp(config.patnum,'full')
    patnum = length(dataset);
else
    patnum = config.patnum;
end


% Set Folder for Trained Networks of according Dataset:
fnL{1}=[pat '/Data/Evaluation/CV/' type];
mkdir(fnL{1})

%% Step 2: Training Loop over Patients

%Set training conditions:
NoInp=config.noinp; 

%Set global hyperparameters (only first one should be set freely rest should stay more or less untouched, i.e. NR unstable):
M= config.numlat;
lam=0; %ALWAYS 0
NR.g=0.0005; %Learning rate of categorical model update via NR (M-step)
NR.bound=0.01; %Convergence bound of categorical NR (M-step)
NR.use=true; %Change to False if PLRNN training with single modality
NR.fixedBeta=0; %Fixes update of categorical model parameters (M-step)

% Set Sub-Folders for Trained Networks according to training conditions:
if NoInp
    fnL{3}=[fnL{1} '/NoInp'];

else
    fnL{3}=[fnL{1} '/Inp'];
end


mkdir(fnL{3});
fnL{4} = [fnL{3} '/m' num2str(M) '/'];
mkdir(fnL{4});

%%
for pt=1:patnum
%CHOOSE patient (data set which we train on)
fMRI=dataset{1,pt};
%%
fnL{1}=['NBKO_PLRNN_dataset_' num2str(pt)];
 
    X_=fMRI.X;
    C_=fMRI.C_;
    Inp=fMRI.Inp;
    for i=1:length(X_)
        Tmtx{1,i}=1:length(X_{i});
    end
    
%% Cross-Validation:
ks=length(X_);%length(knr);
 mu0=cell(1,ks);
 B=cell(1,ks);
 G=cell(1,ks);
 A=cell(1,ks);
 W=cell(1,ks);
 C=cell(1,ks);
 h=cell(1,ks);
 S=cell(1,ks);
 Beta=cell(1,ks);
 LL=cell(1,ks);
 Vest=cell(1,ks);
 Xn=cell(1,ks);
 Ezi=cell(1,ks);
    
    
for k=1:ks
%Pre-define input variables:
   % Define CtrPar/NR struct:
    Cstr=struct;
    Cstr.lam=0;


%Pre-define input variables:
   % Define CtrPar/NR struct:
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
  
   m= config.numlat;
    
   Cstr.Lb=[]; Cstr.Ub=[]; Cstr.XZspl=[];
    %Initialize Parameter Est.:
    Kk=size(C_{1},1);
    K= size(Inp{1},1);
    n=size(X_{1},1);%num obs const for all batches
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
    s=.01;
    InPar.C=a+(b-a).*rand(m,K);
    InPar.S=diag(diag(s*rand(m)));


    T=size(Tmtx{1},2);

    InPar.mu0=rand(m,1);
    InPar.B=a+(b-a).*rand(n,m);
    InPar.G=diag(diag(s*rand(n)));

    InPar.h=a+(b-a).*rand(m,1);
    InPar.Beta=a+(b-a).*rand(m,Kk-1);
    [X__,C__,Tmtx_,Inp_]=cutkCVE(k,X_,C_,Tmtx,Inp);
    if NoInp
        Inp__=cell(1,length(Inp_));
        for nbs=1:length(Inp_)
            Inp__{1,nbs}=zeros(K,T);
        end
        InPar.Inp=Inp__;
    else
        InPar.Inp=Inp_;
    end
    
    InPar.X=X__;
    InPar.C_=C__;
    Cstr.ChunkList=[1 length(X__)+1];
    
    %Save Data with categories
%% Training:
      opt=1; %Input given
      NR.mod=0;
      tau=10^6;
    try

    NR.use=true;
    CtrPar.tol = 1*10^(-2);
    CtrPar.LAR=0;
    Cstr.ChunkList=[1 length(InPar.X)+1];
    if NoInp
        [net,OutPar]=runPLRNN_initproc_20_MC_NS_NI(InPar,CtrPar,Cstr,NR,m,opt,tau);
    else     
        [net,OutPar]=runPLRNN_initproc_20_MC_NS(InPar,CtrPar,Cstr,NR,m,opt,tau);
    end%    
    %Unpack parameter estimates:
    mu0{k}=OutPar.mu0;
    B{k}=OutPar.B;
    G{k}=OutPar.G;
    A{k}=OutPar.A;
    W{k}=OutPar.W;
    C{k}=OutPar.C;
    h{k}=OutPar.h;
    S{k}=OutPar.S;
    Beta{k}=OutPar.Beta;
    LL{k}=OutPar.LL;
    Xn{k}=OutPar.X;
    lam=Cstr.lam;
    Ezi{k}=OutPar.Ezi;
    Lb=Cstr.Lb;
    Ub=Cstr.Ub;
    M=m;
        
    if NoInp
        fnOut{3}=[fnL{4} 'data_sparse_PLRNN_m' num2str(m) 'pat_' num2str(pt) '_cv_' num2str(k) '.mat'];
        save(fnOut{3},'OutPar','X_','C_','Inp','M','lam');
    else
        fnOut{3}=[fnL{4} 'data_sparse_PLRNN_m' num2str(m) 'pat_' num2str(pt) '_cv_' num2str(k) '.mat'];
        save(fnOut{3},'OutPar','X_','C_','Inp','M','lam');
    end
    
catch exc1
    Ezi{k}=exc1;
end

end


end 

end

