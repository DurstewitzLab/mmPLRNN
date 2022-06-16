function [OutPar,EziSv,VestSv,LL]= EMiterLDSmultDS(CtrPar,InPar,Cstr)
%
% please cite (& consult for further details): Durstewitz (2017) PLoS Comp Biol
% 
% implements EM iterations for LDS
% z_t = W z_t-1 + h + C Inp_t + e_t , e_t ~ N(0,S)
% x_t = B z_t + nu_t , nu_t ~ N(0,G)
%
% NOTE: Inp, mu0_est, X can be matrices or cell arrays. If provided as cell
% arrays, each cell is interpreted as a single 'trial'; trials are
% concatenated for estimation, but temporal gaps are assumed between
% trials, i.e. temporal breaks are 'inserted' between trials
%
%
% REQUIRED INPUTS:
% CtrPar=[tol MaxIter __ eps]: vector of control parameters:
% -- tol: relative (to first LL value) tolerated increase in log-likelihood
% -- MaxIter: maximum number of EM iterations allowed
% -- __: 3rd param. is irrelevant for LDS (just included to parallel PLRNN files) 
% -- eps: singularity parameter in StateEstLDS
% A_est: initial estimate of MxM diagonal matrix of auto-regressive weights
% W_est: initial estimate of MxM off-diagonal matrix of interaction weights
% S_est: initial estimate of MxM diagonal process covariance matrix
%        (assumed to be constant here)
% Inp: MxT matrix of external inputs, or cell array of input matrices 
% mu0_est: initial estimate of Mx1 vector of initial values, or cell array of Mx1 vectors
% B_est: initial estimate of NxM matrix of regression weights
% G_est: initial estimate of NxN diagonal observation covariance matrix
% h: Mx1 vector of (fixed) thresholds
% X: NxT matrix of observations, or cell array of observation matrices
%
% OPTIONAL INPUTS:
% XZspl: vector [Nx Mz] which allows to assign certain states only to 
%        certain observations; specifically, the first Mz state var are
%        assigned to the first Nx obs, and the remaining state var to the
%        remaining obs; (1:Mz)-->(1:Nx), (Mz+1:end)-->(Nx+1:end)
%
%
% OUTPUTS:
% final estimates of network parameters {mu0_est,B_est,G_est,W_est,A_est,S_est}
% Ezi: MxT matrix of state expectancies as returned by StateEstLDS
% Vest: estimated state covariance matrix E[zz']-E[z]E[z]'
% LL: log-likelihood (vector) as function of EM iteration 

if isempty(Cstr.Ub), Ub=[]; else Lb=Cstr.Lb; end;
if isempty(Cstr.Lb), Lb=[]; else Ub=Cstr.Ub; end;
if isempty(Cstr.XZspl), XZspl=[]; else XZspl=Cstr.XZspl; end;


%Input(Struct containing all Parameter estimates)
A_est=InPar.A;
W_est=InPar.W;
C_est=InPar.C;
S_est=InPar.S;
Inp_=InPar.Inp;
mu0est=InPar.mu0;
Best=InPar.B;
Gest=InPar.G;
h_est=InPar.h;
X_=InPar.X;
% c=InPar.C_;
if isempty(Cstr.ChunkList) ChunkList=[1 length(X_)+1]; else ChunkList=Cstr.ChunkList; end;


%Make Cells: 
if iscell(X_), X=X_; Inp=Inp_; 
else X{1}=X_; Inp{1}=Inp_; end;
if iscell(Best) B_est=Best; G_est=Gest;
else B_est{1}=Best; G_est{1}=Gest; end;



tol = CtrPar.tol;
MaxIter=CtrPar.MaxIter;
eps=CtrPar.eps;
fixedS=CtrPar.fixedS;   % S to be considered fixed or to be estimated
fixedC=CtrPar.fixedC;   % C to be considered fixed or to be estimated
fixedB=CtrPar.fixedB;   % S to be considered fixed or to be estimated
fixedG=CtrPar.fixedG;   % C to be considered fixed or to be estimated

nbatch=length(X);   % number parallel processes
m=length(h_est);


if iscell(mu0est)
    if length(mu0est)~=nbatch mu0est=cell2mat(mu0est); mu0_est=cell(1,nbatch); for i=1:nbatch mu0_est{i}=mu0est; end; 
    else mu0_est=mu0est; end;
else mu0_est=cell(1,nbatch); for i=1:nbatch mu0_est{i}=mu0est; end; end;

%% EM loop
i=1; LLR=1e8; LL=[]; maxLL=-inf;
Ezi=cell(1,nbatch);
CtrPar.mod=0;
while (i==1 || LLR>tol*abs(LL(1)) || LLR<0) && i<MaxIter

    if fixedB, B0=B_est; else B0=cell(1,length(ChunkList)-1); end
    if fixedG, G0=G_est; else G0=cell(1,length(ChunkList)-1); end
    
    % E-step
    % for sequential updating (NOTE: to reduce memory load, E-matrices may
    % be discarded after each run):
    EV=[];
    for nb=1:nbatch
        
        [j,k]=ismember(nb,ChunkList);
        if j, kk=k; end    % alternative would be to use new {B,G} as soon as updated
        
        [Ezi{nb},Vest]=StateEstLDS2(W_est,C_est,S_est,Inp{nb},mu0_est{nb},B_est{kk},G_est{kk},h_est,X{nb},eps,CtrPar);
        EV=UpdateExpSumsLDSmultDS(Ezi{nb},Vest,X{nb},Inp{nb},EV);
        
        [j,k]=ismember(nb+1,ChunkList);
        if j
            T=cell2mat(cellfun(@size,X(ChunkList(k-1):ChunkList(k)-1),'UniformOutput',false)');
            T=sum(T(:,2));
            [B_est{k-1},G_est{k-1}]=ParEstLDSobs(EV,XZspl,B0{k-1},G0{k-1},T);
            if ChunkList(k)<=nbatch, N=size(X{nb+1},1); 
            EV.E3p=zeros(m); EV.F1=zeros(N,m); EV.F2=zeros(N,N);
            end
        end
    end
    
    % M-step
    if fixedS, S0=S_est; else S0=[]; end
    if fixedC, C0=C_est; else C0=[]; end
    [mu0_est,W_est,S_est,C_est,h_est]=ParEstLDSlat(EV,S0,C0,Lb,Ub);
    
    % compute log-likelihood (alternatively, use ELL output from ParEstLDS)
    EziAll=cell2mat(Ezi);
    LL(i)=LogLikeLDSmultDS(W_est,C_est,S_est,Inp,mu0_est,B_est,G_est,h_est,X,EziAll,ChunkList);
    disp(['LL= ' num2str(LL(i))]);
    
    if i>1, LLR=LL(i)-LL(i-1); else LLR=1e8; end;    % LL ratio 
    if LL(i)>maxLL
        Wsv=W_est; Csv=C_est; Ssv=S_est; mu0sv=mu0_est;
        Bsv=B_est; Gsv=G_est; hsv=h_est; EziSv=Ezi; mu0allSv=mu0_est;
        VestSv=Vest;
    end
    i=i+1;
    
end
disp(['fin LL = ' num2str(LL(end)) ' , #iterations = ' num2str(i-1)]);

OutPar.mu0=mu0sv;
OutPar.B=Bsv;
OutPar.G=Gsv;
OutPar.A=A_est;
OutPar.W=Wsv;
OutPar.C=Csv;
OutPar.h=hsv;
OutPar.S=Ssv;
%%
% (c) 2017 Daniel Durstewitz, Dept. Theoretical Neuroscience,
% Central Institute of Mental Health Mannheim, Heidelberg University
