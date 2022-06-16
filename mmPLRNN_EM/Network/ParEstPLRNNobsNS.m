function [B,G, Beta_est,G_theta]=ParEstPLRNNobsNS(EV,XZsplit,B0,G0,T,Vest,Beta,Cs,NR)
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
% Beta: MxK matrix of latent state parameters(inv. temp) that determine
%       prob. of categorical states c_i
% C_:KxT matrix of categorical states c_i, or cell array of cat. matrices
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
% Beta: MxK matrix containing the beta vector to each category k



eps=1e-5;   % minimum variance allowed for in S and G

E1p=EV.E1p; F1=EV.F1; F2=EV.F2; % none of these occurs in latent model!


%% solve for parameters {B,G} of observation model
if nargin>1 && ~isempty(XZsplit)
    Nx=XZsplit(1); Mz=XZsplit(2);
    F1_=F1; F1_(1:Nx,Mz+1:end)=0; F1_(Nx+1:end,1:Mz)=0;
    E1p_=E1p; E1p_(1:Mz,Mz+1:end)=0; E1p_(Mz+1:end,1:Mz)=0;
    if nargin<3 || isempty(B0), B=F1_*E1p_^-1; else B=B0; end
    if nargin<4 || isempty(G0)
        G=diag(max(diag(F2-F1_*B'-B*F1_'+B*E1p_'*B')./T,eps));   % assumes G to be diag
    else G=G0; end
else
    if nargin<3 || isempty(B0), B=F1*E1p^-1; else B=B0; end
    if nargin<4 || isempty(G0)
        G=diag(max(diag(F2-F1*B'-B*F1'+B*E1p'*B')./T,eps));   % assumes G to be diag
    else G=G0; end
end

%% solve for parameters {beta} of 2. observation model(multi-cat.):

Beta_est=0;
G_theta=0;

if NR.use

    EV.Varz=Vest;
    EV.Ez=EV.Ezi;
    C_=Cs;
    if NR.fixedBeta
        Beta_est=Beta;
    else

    [BetaEst,G_theta]=Full_NR(NR,C_,Beta,EV);

    %% Addding up batch estimates(calculation of chunk est.)
    Beta_est=Beta_est+BetaEst; %Summation of Grad and Hess instead of over estimates!

    
    end
end
    
end
%%
