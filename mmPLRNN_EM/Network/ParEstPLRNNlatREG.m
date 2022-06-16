function [mu0,W,A,S,C,h]=ParEstPLRNNlatREG(EV,S0,C0,Lb,Ub,reg,lam)
if nargin<7, lam=0; else lam=reg.lam; end
%
% 
% implements parameter estimation for PLRNN system
% z_t = A z_t-1 + W max(z_t-1,0) + h + C Inp_t + e_t , e_t ~ N(0,S)
% x_t = B max(z_t,0) + nu_t , nu_t ~ N(0,G)
%
% NOTE: X_, Inp_ can be matrices or cell arrays. If provided as cell
% arrays, each cell is interpreted as a single 'trial'; info is aggregated
% across trials, but temporal gaps are assumed between
% trials, i.e. temporal breaks are 'inserted' between trials
%
%
% REQUIRED INPUTS:
% Ez: MxT matrix of state expectancies as returned by StateEstPLRNN
% V: state covariance matrix E[zz']-E[z]E[z]'
% Ephizi: E[phi(z)]
% Ephizij: E[phi(z) phi(z)']
% Eziphizj: E[z phi(z)']
% X_: NxT matrix of observations, or cell array of observation matrices
% Inp_: KxT matrix of external inputs, or cell array of input matrices 
%
% OPTIONAL INPUTS:
% XZsplit: vector [Nx Mz] which allows to assign certain states only to 
%          certain observations; specifically, the first Mz state var are
%          assigned to the first Nx obs, and the remaining state var to the
%          remaining obs; (1:Mz)-->(1:Nx), (Mz+1:end)-->(Nx+1:end)
% S0: fix process noise-cov matrix to S0
%
% OUTPUTS:
% mu0: Mx1 vector of initial values, or cell array of Mx1 vectors
% B: NxM matrix of regression weights
% G: NxN diagonal covariance matrix
% W: MxM off-diagonal matrix of interaction weights
% A: MxM diagonal matrix of auto-regressive weights
% S: MxM diagonal covariance matrix (Gaussian process noise)
% C: MxK matrix of regression weights multiplying with Kx1 Inp
% h: Mx1 vector of bias terms
% ELL: expected (complete data) log-likelihood


eps=1e-5;   % minimum variance allowed for in S and G

E1=EV.E1; E2=EV.E2; E3=EV.E3; E4=EV.E4; E5=EV.E5; E3_=EV.E3_;
F3=EV.F3; F4=EV.F4; F5_=EV.F5_; F6_=EV.F6_;
f5_1=EV.f5_1; f6_1=EV.f6_1;
Zt1=EV.Zt1; Zt0=EV.Zt0; phiZ=EV.phiZ; InpS=EV.InpS;
T=EV.T;
ntr=EV.ntr;
m=size(E1,1);
Minp=size(InpS,1);

F5=F5_+f5_1;
F6=F6_+f6_1;


%% solve for
% - interaction weight matrix W
% - auto-regressive weights A
% - bias terms h
% - external regressor weights C
% in one go:
I=eye(m);
O=ones(m)-I;

if nargin<4 || isempty(C0)
    Mask=[I;O;ones(1,m);ones(Minp,m)];
    AWhC=zeros(m,m+1+Minp);
    EL=[E3,E4',Zt1,F3';E4,E1+lam*I,phiZ,F4';Zt1',phiZ',T-ntr,InpS';F3,F4,InpS,F6_]; 
    ER=[E2,E5,Zt0,F5_];
else
    Mask=[I;O;ones(1,m)];
    AWhC=zeros(m,m+1);
    
    EL=[E3,E4',Zt1;E4,E1,phiZ;Zt1',phiZ',T-ntr];
    ER=[E2-C0*F3,E5-C0*F4,Zt0-C0*InpS];
end
W=zeros(m);

if nargin<7 || (isempty(Lb) && isempty(Ub))
    
    for i=1:m
        
        [Reg_EL, Reg_ER]=getORegularization(reg,Mask,S0,m,i); %get regularization 
        el=EL+Reg_EL;
        er=ER+Reg_ER;
        
        k=find(Mask(:,i));
        X=el(k,k);
        Y=er(i,k);
        AWhC(i,:)=Y*X^-1;
        W(i,:)=[AWhC(i,2:i) 0 AWhC(i,i+1:m)];
    end
    
    A=diag(AWhC(:,1));
    h=AWhC(:,m+1);
    if nargin<4 || isempty(C0)
        C=AWhC(:,m+2:end);
    else
        C=C0;
    end
    
else

    HH=[]; hh=[];
    for i=1:m
        k=find(Mask(:,i));
        X=EL(k,k);
        Y=ER(i,k);

        HH=blkdiag(HH,X');
        hh=[hh;Y'];
    end
    Lb=Lb'; lb=Lb(1:end)';
    Ub=Ub'; ub=Ub(1:end)';
    awhc=quadprog(HH,-hh,[],[],[],[],lb,ub);
    AWhC=reshape(awhc,size(AWhC,2),m)';
    for i=1:m, W(i,:)=[AWhC(i,2:i) 0 AWhC(i,i+1:m)]; end
    
    A=diag(AWhC(:,1));
    h=AWhC(:,m+1);
    if nargin<4 || isempty(C0)
        C=AWhC(:,m+2:end);
    else
        C=C0;
    end
end



if sum(sum(isnan(C)))>0
    keyboard
    C=C0;
elseif sum(sum(isinf(C)))>0
    keyboard
    C=C0;
end

%% solve for trial-specific parameters mu0
for i=1:ntr
    mu0{i}=EV.AllIni0(:,i)-C*EV.AllInp(:,i);
end



%% solve for process noise covariance S, or use provided fixed S0
if nargin>2 && ~isempty(S0), S=S0;
else
    H=zeros(m);
    for i=1:ntr
        H=H+EV.Ezz0{i}-EV.AllIni0(:,i)*mu0{i}'-mu0{i}*EV.AllIni0(:,i)'+mu0{i}*mu0{i}'+mu0{i}*EV.AllInp(:,i)'*C'+C*EV.AllInp(:,i)*mu0{i}';
    end
    S=diag(diag(H+E3_'-F5*C'-C*F5'+C*F6*C'-E2*A'-A*E2' ...
        +A*E3'*A'-E5*W'-W*E5'+W*(E1+lam*I)'*W'+A*E4'*W'+W*E4*A'+A*F3'*C'+C*F3*A'+W*F4'*C'+C*F4*W' ...
        -Zt0*h'-h*Zt0'+A*Zt1*h'+h*Zt1'*A'+W*phiZ*h'+h*phiZ'*W'+(T-ntr).*h*h'+C*InpS*h'+h*InpS'*C'))./T;   % assumes S to be diag
end

%%

function [Reg_EL, Reg_ER]=getORegularization(reg,Mask,S0,m,irow)

Reg_ER=0;
sigma=unique(diag(S0));

%get mask defining elements to be regularized
%LMask is of size LReg=[A W h C] with -1 for pars->1 and 1 for pars->0
LMask=reg.Lreg;

%0th-order regularization for A/W/h->0
Reg0=0;
tau=reg.tau;
N=size(Mask,1);
if (~isempty(tau) && tau~=0)
    
    I=eye(N);
    tmp=LMask(irow,:);
    ind=find(tmp==1); %find indices of parameters to be regularized
    II=eye(N);
    II(ind,ind)=0;
    
    O1=I-II;     
    Reg0=sigma*tau.*O1;
end

%0th-order regularization for diagonal A with A->1
Reg1=0;
lambda=reg.lambda;
if (~isempty(lambda) && lambda~=0)
    
    I=eye(N);
    tmp=LMask(irow,:);
    ind=find(tmp==-1); %find indices of parameters to be regularized
    II=eye(N);
    II(ind,ind)=0;
    
    O1=I-II;     
    O2=zeros(m,N);
    Aindex=ind(ismember(ind,1:m));
    II=eye(length(Aindex));
    O2(Aindex,Aindex)=II;
    
    

    Reg1=sigma*lambda.*O1;
    Reg_ER=sigma*lambda*O2;
end
Reg_EL=Reg0+Reg1;


