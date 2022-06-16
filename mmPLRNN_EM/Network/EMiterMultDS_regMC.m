function [OutPar,EziSv,VestSv,EphiziSv,EphizijSv,EziphizjSv,LL]= ...
    EMiterMultDS_regMC(CtrPar,InPar,NR,XZspl,Lb,Ub,lam,ChunkList,reg)
% --- this is a parallel implementation based on integration of expectation
% sums across segments!
%
% 
% implements EM iterations for PLRNN system
% z_t = A z_t-1 + W max(z_t-1,0) + h + Inp_t + e_t , e_t ~ N(0,S)
% x_t = B max(z_t,0) + nu_t , nu_t ~ N(0,G)
% c_t ~ MC(P_i({z_t})), P_i = exp(b_i*z_t)/sum_j exp(b_j*z_t)   ||| p(c_t|z_t) = prod_{i=1}^{K} P_i ^c_ti
%
% NOTE: Inp, mu0_est, X can be matrices or cell arrays. If provided as cell
% arrays, each cell is interpreted as a single 'trial'; trials are
% concatenated for estimation, but temporal gaps are assumed between
% trials, i.e. temporal breaks are 'inserted' between trials
%
%
% REQUIRED INPUTS:
% CtrPar= struct/list of control parameters:
% -- tol: relative (to first LL value) tolerated increase in log-likelihood
% -- MaxIter: maximum number of EM iterations allowed
% -- tol2: relative error tolerance in state estimation (see StateEstPLRNN) 
% -- eps: singularity parameter in StateEstPLRNN
% -- flipOnIt: (#itr until flip all off)parameter that controls switch from single (i<=flipOnIt) to 
%              full (i>flipOnIt) constraint flipping in StateEstPLRNN
% -- boundE: boundary condition for the NR solver in the StateEstPLRNN_MC
% -- gE: learning rate for the NR solver in the StateEstPLRNN_MC (mostly 1 = full NR)
% -- outbnd: bound for the outer loop NR solver in the StateEstPLRNN_MC
% -- exp: defines data type of the observations (true==non-synthetic data, false==simulated data)
% NR = struct of parameters for NR in ParEstPLRNNobs:
% -- boundM: boundary condition for the NR solver in the ParEstPLRNNobs
% -- gM: learning rate for the NR solver in the ParEstPLRNNobs (mostly 0.01)
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
% C: KxT matrix of categories
%
% OPTIONAL INPUTS:
% XZspl: vector [Nx Mz] which allows to assign certain states only to 
%        certain observations; specifically, the first Mz state var are
%        assigned to the first Nx obs, and the remaining state var to the
%        remaining obs; (1:Mz)-->(1:Nx), (Mz+1:end)-->(Nx+1:end)
% Lb: lower bounds on W matrix
% Ub: upper bounds on W matrix
%
%
% OUTPUTS:
% final estimates of network parameters {mu0_est,B_est,G_est,W_est,A_est,S_est}
% Ezi: MxT matrix of state expectancies as returned by StateEstPLRNN
% Vest: estimated state covariance matrix E[zz']-E[z]E[z]'
% Ephizi: E[phi(z)]
% Ephizij: E[phi(z) phi(z)']
% Eziphizj: E[z phi(z)']
% LL: log-likelihood (vector) as function of EM iteration 
% Err: final error returned by StateEstPLRNN
% NumIt: total number of EM + mode-search iterations

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
Betaest=InPar.Beta;
X_=InPar.X;
c=InPar.C_;

%Make Cells: 
if iscell(X_), X=X_; Inp=Inp_; C_=c;
else X{1}=X_; Inp{1}=Inp_; C_{1}=c;end;

if iscell(Best) B_est=Best; else B_est{1}=Best; end
if iscell(Betaest) Beta_est=Betaest; else Beta_est{1}=Betaest; end
if iscell(Gest) G_est=Gest; else G_est{1}=Gest; end;


if nargin<4, XZspl=[]; end
if nargin<5, Lb=[]; end
if nargin<6, Ub=[]; end
if nargin<7, lam=0; end
if nargin<8 || isempty(ChunkList), ChunkList=[1 length(X)+1]; end

tol=CtrPar.tol;
MaxIter=CtrPar.MaxIter;
tol2=CtrPar.tol2;
eps=CtrPar.eps;
flipOnIt=CtrPar.flipOnIt;
FinOpt=CtrPar.FinOpt;   % quad. prog. step at end of E-iterations
fixedS=CtrPar.fixedS;   % S to be considered fixed or to be estimated
fixedC=CtrPar.fixedC;   % C to be considered fixed or to be estimated
fixedB=CtrPar.fixedB;   % B to be considered fixed or to be estimated
fixedA=CtrPar.fixedA;   % A to be considered fixed or to be estimated
fixedW=CtrPar.fixedW;   % W to be considered fixed or to be estimated
fixedh=CtrPar.fixedh;   % h to be considered fixed or to be estimated
fixedmu0=CtrPar.fixedmu0;   % mu0 to be considered fixed or to be estimated
Cceil=CtrPar.Cceil; %limitation for the influence of external inputs

% Support of old 'list' style of CtrPar:
if ~isstruct(CtrPar)
    CtrParList = CtrPar;
    CtrPar = struct();
    CtrPar.tol = CtrParList(1);
    CtrPar.MaxIter = CtrParList(2);
    CtrPar.tol2 = CtrParList(3);
    CtrPar.eps = CtrParList(4);
    CtrPar.flipOnIt = CtrParList(5);
    CtrPar.FinOpt = CtrParList(6);   % quad. prog. step at end of E-iterations
    CtrPar.fixedS = CtrParList(7);   % S to be considered fixed or to be estimated
    CtrPar.fixedC = CtrParList(8);   % C to be considered fixed or to be estimated
    CtrPar.fixedB = CtrParList(9);   % B to be considered fixed or to be estimated
    CtrPar.fixedG = CtrParList(10);   % G to be considered fixed or to be estimated
end

%Define Parameters for NR:
outbnd=CtrPar.outbnd;
        
        %Define options for NR:
        options.tol=CtrPar.tol; 
        options.flipAll = false; %will be turned on inside loop
        options.eps = CtrPar.eps; 
        options.FinOpt=CtrPar.FinOpt;
        options.g_min=CtrPar.gE; %learningrate NR
        options.bound =CtrPar.boundE; %boundary condition NR
%         options.exp=CtrPar.exp; %defines data type
        options.maxItr=CtrPar.maxItr;
        options.mod=NR.mod;

if ~isfield(CtrPar, 'getAllEzi'); CtrPar.getAllEzi = false; end
m=length(h_est);


nbatch=length(X);   % number of batches
if iscell(mu0est)
    if length(mu0est)~=nbatch mu0est=cell2mat(mu0est); mu0_est=cell(1,nbatch); for i=1:nbatch mu0_est{i}=mu0est; end; 
    else mu0_est=mu0est; end;
else mu0_est=cell(1,nbatch); for i=1:nbatch mu0_est{i}=mu0est; end; end;

%% EM loop
Vest=cell(1,nbatch);
i=1; LLR=1e8; LL=[]; maxLL=-inf;
while (i==1 || LLR>tol*abs(LL(1)) || LLR<0) && i<CtrPar.MaxIter

    
    if CtrPar.fixedB, B0=B_est; else B0=cell(1,length(ChunkList)-1); end
    if CtrPar.fixedG, G0=G_est; else G0=cell(1,length(ChunkList)-1); end
    
        
    % E-step & M-step obs. -----------------------------------
    

    Ephizi = cell(1, nbatch);
    Ephizij = cell(1, nbatch);
    Eziphizj = cell(1, nbatch);
    U = cell(1, nbatch);
    
    for nb=1:nbatch


        nChunk = find(ChunkList > nb, 1) - 1; % get current chunk number
        z_eval=[];

        LL_path=[];
        LL_itr=[];
        criterion=1000;
        itr=0;
      
        

        while criterion>outbnd && itr<=options.maxItr
        %Define overall convergence criterion by analysing change in final LL after a  
            %clear criterion
            itr=itr+1;
            Ezi{nb}=z_eval;
            d0=[];
            if ~NR.use
                [z_eval,U_bar,d,Err,LLE,Z]=StateEstPLRNN_Gauss(A_est,W_est,C_est,S_est,Inp{nb},mu0_est{nb},B_est{nChunk},G_est{nChunk},h_est,X{nb},Ezi{nb},d0,options);
            else
                [z_eval,d,Err,LLE,Z,U_bar]=StateEstPLRNN_MCs(A_est,W_est,C_est,S_est,Inp{nb},mu0_est{nb},B_est{nChunk},G_est{nChunk},h_est,Beta_est{nChunk},X{nb},C_{nb},Ezi{nb},d0,options);
            end
            if itr==1
                LL_itr=LLE(1); 
            end
            LL_itr=[LL_itr,LLE(end)];
            LL_path=[LL_path,LLE];

            if itr<3
                criterion = abs(LLE(end)-mean(LL_itr));
            else
                criterion = abs(LLE(end)-mean(LL_itr(end-1:end)));
            end
            
        end

        Ezi{nb}=z_eval;
        U{nb}=-U_bar;
    
        [Ephizi{nb}, Ephizij{nb}, Eziphizj{nb}, Vest{nb}] = ...
                                             ExpValPLRNN2c(Ezi{nb},U{nb});
        ExpSums(nb) = GetExpSums(Ezi{nb},Vest{nb},Ephizi{nb},Ephizij{nb},...
                                 Eziphizj{nb},X{nb},Inp{nb});
     end
        


      for nChunk = 1:length(ChunkList)-1
        chunkBatches = ChunkList(nChunk):ChunkList(nChunk+1)-1;
        Xs = X(chunkBatches);
        Ezis=Ezi(chunkBatches);
        Cs = C_(chunkBatches);
        Var=Vest(chunkBatches);
        T = size([Xs{:}], 2);
        EVobs = sumUpObsSums(ExpSums(chunkBatches));
        EVobs.Ezi=Ezis;
            [B_est{nChunk},G_est{nChunk},Beta_est{nChunk},G_theta]=ParEstPLRNNobsNS(EVobs,XZspl,B0{nChunk},G0{nChunk},T,Var,Beta_est{nChunk},Cs,NR);
      end

   
    
    
    % M-step process model -----------------------------------
    if fixedS, S0=S_est; else S0=[]; end
    if fixedC, C0=C_est; else C0=[]; end
    if fixedA, A0=A_est; end
    if fixedW, W0=W_est; end
    if fixedh, h0=h_est; end
    if fixedmu0, mu00=mu0_est; end
    
    EV = sumUpLatSums(ExpSums);
    if CtrPar.LAR
        [mu0_est,W_est,A_est,S_est,C_est,h_est]=ParEstPLRNNlatREG(EV,S0,C0,Lb,Ub,reg,lam);
    else
        [mu0_est,W_est,A_est,S_est,C_est,h_est]=ParEstPLRNNlat(EV,S0,C0,Lb,Ub,lam);
    end
    if fixedA, A_est=A0; end
    if fixedW, W_est=W0; end
    if fixedh, h_est=h0; end
    if fixedmu0, mu0_est=mu00; end
    
    % compute log-likelihood (alternatively, use ELL output from ParEstPLRNN)
    EziAll=cell2mat(Ezi);
    if NR.use
        if CtrPar.LAR
        LL(i)=LogLikePLRNN4_MC(A_est,W_est,C_est,S_est,Inp,mu0_est,B_est,G_est,h_est,Beta_est,X,C_,EziAll,ChunkList,reg);
        else
            LL(i)=LogLikePLRNNmultDS_MC(A_est,W_est,C_est,S_est,Inp,mu0_est,B_est,G_est,h_est,Beta_est,X,C_,EziAll,lam, ChunkList,NR.mod);
        end
    else
            LL(i)=LogLikePLRNNmultDS_Reg(A_est,W_est,C_est,S_est,Inp,mu0_est,B_est,G_est,h_est,X,EziAll,lam,ChunkList,reg);
    end
    disp(['LL= ' num2str(LL(i))]);

    
    if i>1, LLR=LL(i)-LL(i-1); else LLR=1e8; end   % LL ratio 
    
    if LL(i)>maxLL
        Asv=A_est; Wsv=W_est; Csv=C_est; Ssv=S_est; mu0sv=mu0_est;
        Bsv=B_est; Gsv=G_est; hsv=h_est;
        Betasv=Beta_est;
        % Give all Ezi if specified, otherwise give only last one
        if CtrPar.getAllEzi
            EziSv=Ezi;
            VestSv=Vest;
        else
            EziSv=Ezi{end};
            VestSv=Vest{end};
        end
        EphiziSv=Ephizi{end};
        EphizijSv=Ephizij{end}; EziphizjSv=Eziphizj{end};
        maxLL=LL(i);
    else
        
        Asv=A_est; Wsv=W_est; Csv=C_est; Ssv=S_est; mu0sv=mu0_est;
        Bsv=B_est; Gsv=G_est; hsv=h_est;
        Betasv=Beta_est;
        % Give all Ezi if specified, otherwise give only last one
        if CtrPar.getAllEzi
            EziSv=Ezi;
            VestSv=Vest;
        else
            EziSv=Ezi{end};
            VestSv=Vest{end};
        end
        EphiziSv=Ephizi{end};
        EphizijSv=Ephizij{end}; EziphizjSv=Eziphizj{end};
        [M,I]=max(LL);
    end
    i=i+1;
    % Limit the influence of external inputs:
    Cceil=mean(mean(A_est+W_est));
    if ~fixedC
        if mean(mean(C_est)) >= Cceil
            fixedC=1;
        end
    end
end
disp(['LL= ' num2str(LL(end)) ', # iterations= ' num2str(i-1)]);

OutPar.mu0=mu0sv;
OutPar.B=Bsv;
OutPar.G=Gsv;
OutPar.A=Asv;
OutPar.W=Wsv;
OutPar.C=Csv;
OutPar.h=hsv;
OutPar.S=Ssv;
OutPar.Beta=Betasv;
end
function EV = sumUpObsSums(ExpSums)
    % Sum up all sums for observation model estimation
    EV = ExpSums(1);
    if length(ExpSums)>1
        for i = 2:length(ExpSums)
            EV.E1p      = EV.E1p      + ExpSums(i).E1p;
            EV.F1       = EV.F1       + ExpSums(i).F1;
            EV.F2       = EV.F2       + ExpSums(i).F2;
        end
    end
end

function EV = sumUpLatSums(ExpSums)
    % Sum up all sums for latent model estimation
    EV = ExpSums(1);
    if length(ExpSums)>1
        for i = 2:length(ExpSums)
            EV.T        = EV.T        + ExpSums(i).T;
            EV.E1       = EV.E1       + ExpSums(i).E1;
            EV.E2       = EV.E2       + ExpSums(i).E2;
            EV.E3       = EV.E3       + ExpSums(i).E3;
            EV.E4       = EV.E4       + ExpSums(i).E4;
            EV.E5       = EV.E5       + ExpSums(i).E5;
            EV.E3_      = EV.E3_      + ExpSums(i).E3_;
            EV.F3       = EV.F3       + ExpSums(i).F3;
            EV.F4       = EV.F4       + ExpSums(i).F4;
            EV.F5_      = EV.F5_      + ExpSums(i).F5_;
            EV.F6_      = EV.F6_      + ExpSums(i).F6_;
            EV.f5_1     = EV.f5_1     + ExpSums(i).f5_1;
            EV.f6_1     = EV.f6_1     + ExpSums(i).f6_1;
            EV.Zt1      = EV.Zt1      + ExpSums(i).Zt1;
            EV.Zt0      = EV.Zt0      + ExpSums(i).Zt0;
            EV.phiZ     = EV.phiZ     + ExpSums(i).phiZ;
            EV.InpS     = EV.InpS     + ExpSums(i).InpS;
            EV.AllIni0  = [EV.AllIni0, ExpSums(i).AllIni0];
            EV.AllInp   = [EV.AllInp, ExpSums(i).AllInp];
        end
    end
    EV.ntr = length(ExpSums);
    EV.Ezz0 = {ExpSums(:).Ezz0};
end
