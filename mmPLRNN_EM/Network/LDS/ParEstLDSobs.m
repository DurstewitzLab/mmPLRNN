function [B,G]=ParEstLDSobs(EV,XZsplit,B0,G0,T)
%
% please cite (& consult for further details): Durstewitz (2017) PLoS Comp Biol
% 
% implements parameter estimation for LDS system
% z_t = W z_t-1 + h + C Inp_t + e_t , e_t ~ N(0,S), W=A+W
% x_t = B z_t + nu_t , nu_t ~ N(0,G)
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
% W: MxM matrix of interaction weights
% S: MxM diagonal covariance matrix (Gaussian process noise)
% C: MxK matrix of regression weights multiplying with Kx1 Inp
% h: Mx1 vector of bias terms
% ELL: expected (complete data) log-likelihood


eps=1e-5;   % minimum variance allowed for in S and G

E3p=EV.E3p;
F1=EV.F1; F2=EV.F2;


%% solve for parameters {B,G} of observation model
if nargin>1 && ~isempty(XZsplit)
    Nx=XZsplit(1); Mz=XZsplit(2);
    F1_=F1; F1_(1:Nx,Mz+1:end)=0; F1_(Nx+1:end,1:Mz)=0;
    E3p_=E3p; E3p_(1:Mz,Mz+1:end)=0; E3p_(Mz+1:end,1:Mz)=0;
    if nargin<3 || isempty(B0), B=F1_*E3p_^-1; else B=B0; end
    if nargin<4 || isempty(G0)
        G=diag(max(diag(F2-F1_*B'-B*F1_'+B*E3p_'*B')./sum(T),eps));   % assumes G to be diag
    else G=G0; end
else
    if nargin<3 || isempty(B0), B=F1*E3p^-1; else B=B0; end
    if nargin<4 || isempty(G0)
        G=diag(max(diag(F2-F1*B'-B*F1'+B*E3p'*B')./sum(T),eps));   % assumes G to be diag
    else G=G0; end
end


%%
% (c) 2017 Daniel Durstewitz, Dept. Theoretical Neuroscience, Central
% Institute of Mental Health, Heidelberg University
