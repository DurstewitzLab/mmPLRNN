function [LL, LL_pxz, LL_pz]=LogLikePLRNN4_MC(A,W,C,S,Inp_,mu0_,B,G,h,Beta,X_,C_,Z_,ChunkList,reg)


%edits on regularization **** 2019/08
if nargin>11
    
    tau=reg.tau;
    lambda=reg.lambda;
    
    L=[A W h]; 
    LMask=reg.Lreg;
    
    %regularization on pars->0
    Lhat=L.*(LMask==1);
    Reg1=-(tau/2)*trace(Lhat'*Lhat);
    
    %regularization on pars->1
    Lhat=L.*(LMask==-1);
    OneMtx=(LMask==-1).*(L~=0);
    Reg2=-(lambda/2)*trace((Lhat-OneMtx)'*(Lhat-OneMtx));
else
    Reg1=0; Reg2=0;
end

% *** Same as LogLikePLRNN, except that additional matrix C of weights for
% external regressors (i.e., inputs) is included
%
%
% please cite (& consult for further details): ******
%
% log-likelihood for PLRNN system
% z_t = A z_t-1 + W max(z_t-1,0) + h + C Inp_t + e_t , e_t ~ N(0,S)
% x_t = B max(z_t,0) + nu_t , nu_t ~ N(0,G)
%
% NOTE: Inp_, mu0_, X_ can be matrices or cell arrays. If provided as cell
% arrays, each cell is interpreted as a single 'trial'; trials are
% concatenated for estimation, but temporal gaps are assumed between
% trials, i.e. temporal breaks are 'inserted' between trials
%
%
% REQUIRED INPUTS:
% A: MxM diagonal matrix of auto-regressive weights
% W: MxM off-diagonal matrix of interaction weights
% C: MxK matrix of regression weights multiplying with Inp
% S: MxM diagonal process covariance matrix
% Inp_: MxT matrix of external inputs, or cell array of input matrices
% mu0_: Mx1 vector of initial values, or cell array of Mx1 vectors
% B: NxM matrix of regression weights
% G: NxN diagonal observation covariance matrix
% h: Mx1 vector of (fixed) thresholds
% X_: NxT matrix of observations, or cell array of observation matrices
% Z_: MxT matrix of state expectancies as returned by StateEstPLRNN
%
% OUTPUTS:
% LL: model log-likelihood p(X,Z|param)

if iscell(X_), X=X_; Inp=Inp_; mu0=mu0_; Cs=C_;
    T=cell2mat(cellfun(@size,X,'UniformOutput',false)'); T=T(:,2)';
    m=length(mu0{1});
    Z=mat2cell(Z_,m,T);
else X{1}=X_; Inp{1}=Inp_; mu0{1}=mu0_; Z{1}=Z_; Cs{1}=C_; [~,T]=size(X_); m=length(mu0{1});
end
ntr=length(X);


LL=0;
for i=1:ntr   % cycle through all trials
    H=repmat(h,1,T(i));
    P=Z{i}(:,1)-mu0{i}-C*Inp{i}(:,1); LL0=P'*S^-1*P;  % from initial state
    P=Z{i}(:,2:T(i))-A*Z{i}(:,1:T(i)-1)-W*max(0,Z{i}(:,1:T(i)-1))-H(:,1:T(i)-1)-C*Inp{i}(:,2:T(i));
    LL1=trace(P'*S^-1*P);
    k=find(i>=ChunkList,1,'last');
    P=X{i}-B{k}*max(0,Z{i}); LL2=trace(P'*G{k}^-1*P);   % from observation model for that trial
    ll_mc=LL_MC(Beta, Z{i}, T(i), m, Cs{i});
    LL3=2*sum(ll_mc);   % from cat. model for that trial
    LL=LL-1/2*(LL0+LL1+LL2+LL3+T(i)*sum(log(diag(G{k})))+T(i)*sum(log(diag(S))))+Reg1+Reg2;  % works only for diag S,G
end
LL_pxz=LL2;
LL_pz=LL0+LL1;
end

%%
% (c) ******
function [ll] = LL_MC(Bet, z, T, m, c)
%Function of derivative corresponding to MC dist.
% C is a vector of size Tx1
% C_dot is a vector of size MTx1
    
    z=reshape(z,m*T,1);
    C = zeros(T,1);
    beta_star = zeros(T,1);
    [K,T]=size(c);
    b_zt = zeros(K,1);
    C_dot=[];
    if iscell(Bet), Bet=cell2mat(Bet) 
    end;
    
    %calculate C & C_ddot entry by entry:
    for i=1:T
        z_t = z(((i-1)*m+1):i*m);
        c_t=c(:,i);
        if sum(c_t)==0
        
           C(i)=0;
           C_dot=[C_dot;zeros(m,1)];
        else
            temp=sum(exp(z_t'*Bet));
            gamma_zt = log(1+temp);
            C(i)=gamma_zt;
        end
    
        % Calculation of beta_star:
        b_zt(1:end-1)= (z_t'*Bet)';
        b_zt(end)=0;
        beta_star(i)=c_t'*b_zt;
    end
    
    ll=C-beta_star;
end
