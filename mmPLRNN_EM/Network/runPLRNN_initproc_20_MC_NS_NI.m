function [net,OutPar]=runPLRNN_initproc_20_MC_NS_NI(InPar,CtrPar,Cstr,NR,p,opt,tau)

if nargin<5, Cstr.lam=0; end %regularization parameter!

%--------------------------------------------------------------------------
%opt:   1 PLRNN, 2=LRNN
%kset:  index of trials to be left out
%p:     latent state dim
%opt:   1=estimate without inputs; 2= with inputs
%--------------------------------------------------------------------------




M=p;

if iscell(InPar.X), Ym=InPar.X{1}; else Ym=InPar.X; end;

CtrPar.tol=1e-3;   % relative (to first LL value) tolerated increase in log-likelihood
CtrPar.MaxIter=100;    % maximum number of EM iterations allowed

CtrPar.eps=1e-5;   % singularity parameter in StateEstPLRNN
CtrPar.fixedS=1;   % S to be considered fixed or to be estimated

if opt==1
    CtrPar.fixedC=1;
else 
    CtrPar.fixedC=0;
end

CtrPar.fixedB=0;   % B to be considered fixed or to be estimated
CtrPar.fixedG=0;

CtrPar.Cceil=2; %Upper bound of external input influence 

reg.lam=Cstr.lam;
%---------------------------- MAIN ---------------------------------------%
%% --- 1st ini step by LDS; B,G free to vary, S~G
%--------------------------------------------------------------------------


CtrPar.tol=1e-3;   % relative (to first LL value) tolerated increase in log-likelihood
CtrPar.MaxIter=10;    % maximum number of EM iterations allowed

CtrPar.eps=1e-5;   % singularity parameter in StateEstPLRNN
CtrPar.fixedS=1;   % S to be considered fixed or to be estimated

if opt==1
    CtrPar.fixedC=1;
else 
    CtrPar.fixedC=0;
end

CtrPar.fixedB=0;   % B to be considered fixed or to be estimated
CtrPar.fixedG=0;


S=eye(M);
G=diag(var(Ym'));
InPar.S=S;
InPar.G=G;  % take data var as initial estim (~1 here)



[OutPar,EziSv,VestSv,LL1]= EMiterLDSmultDS(CtrPar,InPar,Cstr);
OutPar.Ezi=EziSv;
net.netI0=OutPar;

%% --- 1st step: estimate PLRNN model with B,G free to vary, S~G
%--------------------------------------------------------------------------
A1=OutPar.A;
W1=OutPar.W;
h1=OutPar.h;
% specify regularization
%A and/or W->1 regulated by lambda here set tau, A and/or W->0 regulated by tau
LMask=zeros(size([A1 W1 h1]));

%pars->1
Aind=1:ceil(M/2);    %regularize half of the states with A->1;
LMask(Aind,1:M)=-1;

%pars->0
Wind=1:ceil(M/2);       %which half of the states with W->0;
LMask(Wind,M+1:2*M)=1;
hind=1:ceil(M/2);
LMask(hind,2*M+1)=1;

reg.Lreg=LMask;
reg.lambda=tau; %specifies strength of regularization on pars->1
reg.tau=tau;   %specifies strength of regularization on pars->0

OutPar.S=eye(M);
OutPar.Inp=InPar.Inp;
OutPar.X=InPar.X;
OutPar.C_=InPar.C_;
OutPar.Beta=InPar.Beta;

CtrPar.fixedB=0;
CtrPar.fixedG=0;% G to be considered fixed or to be estimated
% fixing tonly C(external input):
CtrPar.fixedh=0;
CtrPar.fixedW=0;
CtrPar.fixedA=0;
CtrPar.fixedmu0=0;
CtrPar.fixedC=1;
CtrPar.MaxIter=31;
OutPar.S=eye(M);
OutPar.G=diag(var(Ym'));

CtrPar.tol2=1e-3;  % relative error tolerance in state estimation (see StateEstPLRNN)
CtrPar.flipOnIt=10; % parameter that controls switch from single (i<=flipOnIt) to
CtrPar.FinOpt=0;   % quad. prog. step at end of E-iterations

XZspl=Cstr.XZspl; Lb=Cstr.Lb; Ub=Cstr.Ub; lam=Cstr.lam; ChunkList=Cstr.ChunkList;
[OutPar,Ezi,Vest,Ephizi,Ephizij,Eziphizj,LL]=EMiterMultDS_regMC(CtrPar,OutPar,NR,XZspl,Lb,Ub,lam,ChunkList,reg);
OutPar.Ezi=Ezi;
OutPar.LL=LL;
net.net1=OutPar;
disp('second iteration done')


%% --- 2nd step: estimate PLRNN model with B fixed & smaller S<G
%--------------------------------------------------------------------------
OutPar.S=0.1*eye(M); %0.8hera-RAMT, 1vegaJames, 0.9vegaRAMT
OutPar.Inp=InPar.Inp;
OutPar.X=InPar.X;
OutPar.C_=InPar.C_;
CtrPar.getAllEzi=1;
CtrPar.MaxIter=51;
CtrPar.fixedB=1;
CtrPar.fixedG=1;
CtrPar.fixedC=1;
NR.fixedBeta=1; %also fixes Beta


[OutPar,Ezi,Vest,~,Ephizij,Eziphizj,LL]=EMiterMultDS_regMC(CtrPar,OutPar,NR,XZspl,Lb,Ub,lam,ChunkList,reg);
OutPar.Ezi=Ezi;
net.net2=OutPar;
disp('third iteration done')


%% --- 3rd step: estimate PLRNN model with B fixed & smaller S<<G
%--------------------------------------------------------------------------
OutPar.S=0.01*eye(M);
OutPar.Inp=InPar.Inp;
OutPar.X=InPar.X;
OutPar.C_=InPar.C_;
CtrPar.MaxIter=31;
CtrPar.getAllEzi=1;
CtrPar.MaxIter=101; %51
CtrPar.fixedB=1;
CtrPar.fixedG=1;
CtrPar.fixedC=1;
NR.fixedBeta=1; %also fixes Beta


[OutPar,Ezi,Vest,Ephizi,Ephizij,Eziphizj,LL]=EMiterMultDS_regMC(CtrPar,OutPar,NR,XZspl,Lb,Ub,lam,ChunkList,reg);
OutPar.Ezi=Ezi;
net.net3=OutPar;
disp('fourth iteration done')





%--------------------------------------------------------------------------
%save away
OutPar.Inp=InPar.Inp;
OutPar.X=InPar.X;
OutPar.C_=InPar.C_;
OutPar.LL=LL;
OutPar.Ezi=Ezi;
OutPar.Ephizi=Ephizi;
OutPar.Ephizij=Ephizij;
OutPar.Eziphizj=Eziphizj;
OutPar.Vest=Vest;

end


