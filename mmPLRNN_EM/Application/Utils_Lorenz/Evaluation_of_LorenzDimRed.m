function [pat2, run3]=Evaluation_of_LorenzDimRed(sysnr,pat,pat2)
%% Collecting Statistics for the full EM Algo:
%Contains code for statistical anlysis of the full EM with the option of
%categorical observations
% Philine Bommer (14.05.19)
% clear all
% close all
% clc


%% Choose dimension reduction
d = 4;

r_d = 0;
pato = pat;    
path(path,pat);


%% Pre-define Statistic collector():

%Cells with Init Conditions:
Betainit=cell(size(sysnr,2) ,1);
Binit=cell(size(sysnr,2) ,1);
Cinit=cell(size(sysnr,2) ,1);
Gammainit=cell(size(sysnr,2) ,1);
Sigmainit=cell(size(sysnr,2) ,1);
hinit=cell(size(sysnr,2) ,1);
Mu0init=cell(size(sysnr,2) ,1);
Ainit=cell(size(sysnr,2) ,1);
Xinit=cell(size(sysnr,2) ,1);
Winit=cell(size(sysnr,2) ,1);

%Cells with Init Conditions:
BetaEst=cell(size(sysnr,2) ,1);
BEst=cell(size(sysnr,2) ,1);
CEst=cell(size(sysnr,2) ,1);
GammaEst=cell(size(sysnr,2) ,1);
SigmaEst=cell(size(sysnr,2) ,1);
hEst=cell(size(sysnr,2) ,1);
Mu0Est=cell(size(sysnr,2) ,1);
AEst=cell(size(sysnr,2) ,1);
WEst=cell(size(sysnr,2) ,1);

LLAll=cell(size(sysnr,2) ,1);
Ez=cell(size(sysnr,2),1);



%Cells with Gauss-only Estimates:
BetaGauss=cell(size(sysnr,2) ,1);
BGauss=cell(size(sysnr,2) ,1);
CGauss=cell(size(sysnr,2) ,1);
GammGauss=cell(size(sysnr,2) ,1);
SigmaGauss=cell(size(sysnr,2) ,1);
hGauss=cell(size(sysnr,2) ,1);
Mu0Gauss=cell(size(sysnr,2) ,1);
AGauss=cell(size(sysnr,2) ,1);
WGauss=cell(size(sysnr,2) ,1);

LLAllGauss=cell(size(sysnr,2) ,1);
EzGauss=cell(size(sysnr,2),1);

Net_evo=cell(size(sysnr,2) ,1);
Net_evoG=cell(size(sysnr,2) ,1);

%Set Inp to 0 and fix C
Cstr.Lb=[]; Cstr.Ub=[]; Cstr.XZspl=[]; Cstr.lam=0;

nst=size(sysnr,2);

Ntraj=1;
T=1000;
s2=0.001;
sObs=0.001;

parfor nr=1:nst 
    nrs=sysnr(nr);
    rng(nrs,'twister') %seed random num. generator
    %% Loading:
    if nrs<10
        str=['lorenz_traj_chaos_n' num2str(Ntraj) '_T' num2str(1000) '_' num2str(s2) '_0' num2str(nrs) '_' num2str(sObs) '.mat'];
    else
        str=['lorenz_traj_chaos_n' num2str(Ntraj) '_T' num2str(1000) '_' num2str(s2) '_' num2str(nrs) '_' num2str(sObs) '.mat'];
    end
    filename=string(strcat(pato,str));
    syst=load(filename);
    
    
    m=15; %latent space dimension

    %% Dimensionality reduction:
    if d==4
        if r_d==1
            di=3; 
        else
            di=2;
        end
    end
    X_red=syst.Xtrans;
    X_red(di,:)=[];
    n=size(X_red,1);
    %% Pre-define input variables:
    
    %Preset const. Structs: 
    Cstr=struct;
    Cstr.Lb=[]; Cstr.Ub=[]; Cstr.XZspl=[]; Cstr.lam=0;
    
    NR=struct;
    NR.g=0.001;
    NR.bound=0.01;
    NR.use=true;
    NR.fixedBeta=0;
    
   % Define CtrPar/NR struct:
   CtrPar=struct;
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
   CtrPar.fixedC = true;   % C to be considered fixed or to be estimated
   CtrPar.fixedB = false;   % B to be considered fixed or to be estimated
   CtrPar.fixedG= false;

    %% Define InPar struct:
    k=size(syst.C,1);
    q=20;
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
    InPar.C=a+(b-a).*rand(m,q);
    InPar.S=diag(diag(s*rand(m)));

    InPar.Inp=zeros(q,T); %no input
    InPar.mu0=rand(m,1);
    InPar.B=a+(b-a).*rand(n,m);
    InPar.G=diag(diag(s*rand(n)));

    InPar.h=a+(b-a).*rand(m,1);
    InPar.Beta=a+(b-a).*rand(m,k-1);
    InPar.X=X_red;
    InPar.C_=syst.Ctrans;
    Cstr.ChunkList=[];
    
    %Safe Init parameters:
    Mu0init{nr}=InPar.mu0;
    Binit{nr}=InPar.B;
    Gammainit{nr}=InPar.G;
    Ainit{nr}=InPar.A;
    Winit{nr}=InPar.W;
    Cinit{nr}=InPar.C;
    hinit{nr}=InPar.h;
    Sigmainit{nr}=InPar.S;
    Betainit{nr}=InPar.Beta;
    Xinit{nr}=X_red;


    %% Training Algorithm:
    try
    NR.use=true;
    NR.mod=0;    
    CtrPar.tol = 1*10^(-2);
    CtrPar.LAR=0;
    
    opt=1;
    tau=10^7/T; %Regularization
    [net,OutPar]=runPLRNN_initproc_20_MC_NS_NI(InPar,CtrPar,Cstr,NR,m,opt,tau); %Optimized for experimental data, NOTE: Regularization NOT used here


    %Safe:
    LLAll{nr}=OutPar.LL;


    Ez{nr}=OutPar.Ezi;
    
    Net_evo{nr}=net;

    
    %Safe param. estimates:
    Mu0Est{nr}=OutPar.mu0;
    BEst{nr}=OutPar.B;
    GammaEst{nr}=OutPar.G;
    AEst{nr}=OutPar.A;
    WEst{nr}=OutPar.W;
    CEst{nr}=OutPar.C;
    hEst{nr}=OutPar.h;
    SigmaEst{nr}=OutPar.S;
    BetaEst{nr}=OutPar.Beta;
    
    %% Gauss only with full dimensionality:
    NR.use=false;
    %InPar.X=syst.Xunnoisetrans;
    %Mstep and IN and OUTput !!!!!
        [net,OutPar]=runPLRNN_initproc_20_MC_NS_NI(InPar,CtrPar,Cstr,NR,m,opt,tau); %Optimized for experimental data, NOTE: Regularization NOT used here

    %Safe:
    LLAllGauss{nr}=OutPar_Gauss.LL;



    EzGauss{nr}=OutPar_Gauss.Ezi;
    
    Net_evoG{nr}=net_Gauss;
    
    BetaGauss{nr}=OutPar_Gauss.Beta;
    BGauss{nr}=OutPar_Gauss.B;
    CGauss{nr}=OutPar_Gauss.C;
    GammGauss{nr}=OutPar_Gauss.G;
    SigmaGauss{nr}=OutPar_Gauss.S;
    hGauss{nr}=OutPar_Gauss.h;
    Mu0Gauss{nr}=OutPar_Gauss.mu0;
    AGauss{nr}=OutPar_Gauss.A;
    WGauss{nr}=OutPar_Gauss.W;
%% Save:    
filename=['ReducedDimLorenzsystem_3dim_statistics_FullEM_sysnr_',num2str(sysnr(nr)), '.mat'];
InPar.Xinit=syst.Xtrans;
parsave_Lorenz([pat2 filename],OutPar,OutPar_Gauss,InPar)
    catch exc
        %leave cell entry empty:
        sysLL(nr,:)=[nan,nan];
        Ez{nr}=exc;
        EzGauss{nr}=exc;
        disp(exc)
    end

end
run3=true;
%%

filename=['ReducedDimLorenzsystem_3dim_statistics_FullEM_full_sysnr_',num2str(sysnr(end)),'.mat'];
 

save([pat2 filename])
end