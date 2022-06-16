function [z,U,d,Err,LogLike,Z]=StateEstPLRNN_Gauss(A,W,C,S,Inp_,mu0_,B,G,h,X_,z0,d0,options)
%
% 
% implements state estimation for PLRNN system
% z_t = A z_t-1 + W max(z_t-1,0) + h + C Inp_t + e_t , e_t ~ N(0,S)
% x_t = B max(z_t,0) + nu_t , nu_t ~ N(0,G)
% c_t ~ MC(P_i({z_t})), P_i = exp(b_i*z_t)/sum_j exp(b_j*z_t)   ||| p(c_t|z_t) = prod_{i=1}^{K} P_i ^c_ti
%
% NOTE: Inp_, mu0_, X_ can be matrices or cell arrays. If provided as cell
% arrays, each cell is interpreted as a single 'trial'; trials are
% concatenated for estimation, but temporal gaps are assumed between
% trials, i.e. temporal breaks are 'inserted' between trials
%
%
% REQUIRED INPUTS:
% A: MxM diagonal matrix 
% W: MxM off-diagonal matrix
% C: MxK matrix of regression weights multiplying with Inp
% S: MxM diagonal covariance matrix (Gaussian process noise)
% Inp_: KxT matrix of external inputs, or cell array of input matrices 
% mu0_: Mx1 vector of initial values, or cell array of Mx1 vectors
% B: NxM matrix of regression weights
% G: NxN diagonal covariance matrix
% h: Mx1 vector of thresholds
% beta: MTx1 categorcial parameters
% X_: NxT matrix of observations, or cell array of observation matrices
% C_:KxT matrix of categorical observations, or cell array of observation 
%
% OPTIONAL INPUTS:
% z0: initial guess of state estimates provided as (MxT)x1 vector
% d0: initial guess of constraint settings provided as 1x(MxT) vector 
% tol: acceptable relative tolerance for error increases (default: 1e-2)
% eps: small number added to state covariance matrix for avoiding
%      singularities (default: 0)
% flipAll: flag which determines whether all constraints are flipped at
%          once on each iteration (=true) or whether only the most violating
%          constraint is flipped on each iteration (=false)
%
% OUTPUTS:
% z: estimated state expectations
% U: Hessian
% Err: final total threshold violation error
% d: final set of constraints (ie, for which z>h) 

% set options:
FinOpt=options.FinOpt;
tol=options.tol;
eps=options.eps;
flipAll=options.flipAll;



if isempty(options.tol), tol=1e-2; end;
if isempty(options.eps), eps=[]; end;
if isempty(options.flipAll), flipAll=false; end;
if isempty(options.FinOpt), FinOpt=0; end;

m=length(A);    % # of latent states

if iscell(X_), X=X_; Inp=Inp_; mu0=mu0_;
else X{1}=X_; Inp{1}=Inp_; mu0{1}=mu0_; end;
ntr=length(X);  % # of distinct trials

%% construct block-banded components of Hessian U0, U1, U2, and 
u0=S^-1+A'*S^-1*A; K0=-A'*S^-1;
Ginv=G^-1;
u2A=W'*S^-1*W; u2B=B'*Ginv*B; u2=u2A+u2B;
u1=W'*S^-1*A; K2=-W'*S^-1;
U0=[]; U2=[]; U1=[];
v0=[]; v1=[];
Tsum=0;

    
for i=1:ntr   % acknowledge temporal breaks between trials
    T=size(X{i},2); Tsum=Tsum+T;
    U0_ = repBlkDiag(u0,T);
    KK0 = repBlkDiag(K0,T);

    X0=X{i};
    tm=find(sum(isnan(X0)));
    
    if isempty(tm)
        U2_ = repBlkDiag(u2,T);
    else
        if options.mod
            U2_ = BlkDiagU2(u2A,0,T,tm);
        else
            U2_ = BlkDiagU2(u2A,u2B,T,tm);  % removes whole time points tm with missing values
        end
    end
    
    U1_ = repBlkDiag(u1,T);
    KK2 = repBlkDiag(K2,T);
    
    KK0=blkdiag(KK0,K0);
    kk=(T-1)*m+1:T*m;
    if options.mod
        U0_(kk,kk)=S^-1+B'*G^-1*B;
    else
        U0_(kk,kk)=S^-1;
    end
    U0_=U0_+KK0(m+1:end,1:T*m);
    KK0=KK0'; U0_=U0_+KK0(1:T*m,m+1:end);
    if options.mod
        U2_(kk,kk)=0;
    else
        U2_(kk,kk)=B'*G^-1*B;
    end
    U1_(kk,kk)=0; KK2=blkdiag(KK2,K2);
    U1_=U1_+KK2(m+1:end,1:T*m);
    U0=sparse(blkdiag(U0,U0_)); U2=sparse(blkdiag(U2,U2_)); U1=sparse(blkdiag(U1,U1_));
    
    I=C*Inp{i}+repmat(h,1,T);
    vka=S^-1*I; vka(:,1)=vka(:,1)+S^-1*(mu0{i}-h); vkb=A'*S^-1*I(:,2:T);
    if options.mod
        vkc=B'*G^-1*X0;
        v0_=(vka(1:end)+vkc(1:end)-[vkb(1:end) zeros(1,m)])'; v0=[v0;v0_];
    else
        v0_=(vka(1:end)-[vkb(1:end) zeros(1,m)])'; v0=[v0;v0_];
    end
    X0(:,tm)=0; % zero out time points with missing values completely;
    % to zero out only individ. components, for each component xit=nan the
    % i-th row of B has to be set to 0, ie corresp. columns of vka need to
    % be computed such that all rows of B and xt corresp. to missing val.
    % are =0.
    vka=B'*G^-1*X0;
    vkb=-W'*S^-1*I(:,2:T);
    if options.mod
        v1_=([vkb(1:end) zeros(1,m)])'; v1=[v1;v1_];
    else
        v1_=(vka(1:end)+[vkb(1:end) zeros(1,m)])'; v1=[v1;v1_];
    end
end;


%% initialize states and constraint vector
n=1; idx=0; k=[];
if ~isempty(z0), z=z0(1:end)'; 
elseif isempty(z0), z=randn(m*Tsum,1); 
else z=randn(m*Tsum,1); 
end
if ~isempty(d0), d=d0; 
elseif isempty(d0), d=zeros(1,m*Tsum); d(z>0)=1; 
else d=zeros(1,m*Tsum); d(z>0)=1; 
end
Err=1e16;
y=rand(m*Tsum,1); LL=d*y;  % define arbitrary projection vector for detecting already visited solutions 
U=[]; options.dErr=-1e8;



%% mode search via NR: 
    
    % save init step:
    zsv=z; Usv=U; dsv=d;
    
    % (1) solve for states Z given constraints d
    D=spdiags(d',0,m*Tsum,m*Tsum);
    H=D*U1; U=U0+D*U2*D+H+H'; 
    if ~isempty(eps), U=U+eps*speye(size(U)); end;  % avoid singularities 
    
    %Setup input:
    Input.Usv=Usv;
    Input.U0=U0;
    Input.U1=U1;
    Input.U2=U2;
    Input.v0=v0;
    Input.v1=v1;
    Input.dsv=dsv;
    Input.T=T;
    Input.Tsum=Tsum;
    Input.m=m;
    Input.y=y;
    Input.k=k;
    Input.idx=idx;
    
    
    %Semi-NR:
     [ z,n,LogLike,Z,U_bar ] = Semi_NR_GausV2(Input, zsv, options);
     
    U_bar=-U_bar;
%% perform a final constrained optim step
z=reshape(z,m,Tsum);


%%


function [ BigM ] = repBlkDiag( M, number_rep )
% repeats the Matrix M NUMBER_REP times in the block diagonal

MCell = repmat({M}, 1, number_rep);
BigM = blkdiag(MCell{:});

end



    % block-diag matrix with missing observations
    % --- excluding only whole time points in tm
    function U2m=BlkDiagU2(u2A,u2B,T,tm)
        U2m=[];
        for t=1:T
            if ismember(t,tm), U2m=blkdiag(U2m,u2A);
            else U2m=blkdiag(U2m,u2A+u2B); end
        end
    end


end
