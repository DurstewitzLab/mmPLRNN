function [KL,KLGauss,KL2,KLGauss2,d_noutG,d_nout]=add_KLDiv_01(dat,v0,red)
%GK 21/11/2018
%uses statespace3Dhist.m, runsys.m, getFileDetails

%load file
%--------------------------------------------------------------------------
% disp(filei)
% dat=load([pat_data filei]);


%->1) let free system run once
%--------------------------------------------------------------------------
r=28; s=10; b=8/3;                          %parameters of Lorenz system
dt=.02;                                     %spacing used for Lorenz system
t_vec=0:dt:2010-0.02; T=length(t_vec);
opt=odeset('RelTol',1e-5,'AbsTol',1e-8);
[t,v]=ode23(@LorenzEqns,t_vec,v0,opt,r,s,b); % unperturbed Lorenz system
% tmp=(v-mean(v))./std(v); tmp=tmp';
% tmp=v';
tmp=(v-repmat(mean(v), [size(v,1),1]))./(v-repmat(std(v), [size(v,1),1])); tmp=tmp';
burnin=1:500;
tmp(:,burnin)=[];
tX=tmp;
if size(red,2)>1
    d=red(2);
    tmp(d,:)=[];
end
% ------------------------------------------------------------------------

%->2) get model parameters and run inferred system
%----------------------------------------------------------------------
% eval(['A=dat.A' num2str(ext) ';'])
% eval(['W=dat.W' num2str(ext) ';'])
% eval(['h=dat.h' num2str(ext) ';'])
% eval(['B=dat.B' num2str(ext) ';'])
% eval(['mu0_all=dat.mu0' num2str(ext) ';'])
%MC network:
A=dat.AEst;
W=dat.WEst;
C=dat.CEst;
h=dat.hEst;
B=dat.BEst{1};
mu0=dat.mu0Est;
% M=size(A,1);
q=20;
Inp=zeros(q,T);
% mu0=mu0_all{1};
[x,ztmp]=runsys(A,W,h,C,mu0,Inp,B,T);
x(:,burnin)=[];

% x=(x-repmat(mean(x), [size(x,1),1]))./(x-repmat(std(x), [size(x,1),1]));

%Gauss network:
A_g=dat.AGauss;
W_g=dat.WGauss;
C_g=dat.CGauss;
h_g=dat.hGauss;
B_g=dat.BGauss{1};
mu0_g=dat.mu0Gauss;

[xG,ztmpG]=runsys(A_g,W_g,h_g,C_g,mu0_g,Inp,B_g,T);
xG(:,burnin)=[];

% xG=(xG-repmat(mean(xG), [size(xG,1),1]))./(xG-repmat(std(xG), [size(xG,1),1]));

dX=x;
dXG=xG;
%--------------------------------------------------------------------------

%->3) compare true and estimated
%--------------------------------------------------------------------------

%@Leo: 
%tX = normalisierte Trajektorie des wahren Systems
%dX = Trajektorie des rekonstruierten Systems

% measure 1: KL divergence in binned 3D space
%----------------------------------------------------------------------
minx=-4; maxx=4; miny=-4; maxy=4; minz=-4; maxz=4;%boundaries for N(0,1)

% minx=-20; maxx=20; miny=-25; maxy=25; minz=0; maxz=50; %boundaries for N(mu,Sigma)
ss=1;                                              %step size of space
dimx=length(minx:ss:maxx)-1;
dimy=length(miny:ss:maxy)-1;
dimz=length(minz:ss:maxz)-1;

if size(red,2)>1
   [t_h,t_nout]=statespace2Dhist(minx,miny,maxx,maxy,ss,tX);%(minx,miny,minz,maxx,maxy,maxz,ss,tX);%,h1,'r');
   [d_h,d_nout]=statespace2Dhist(minx,miny,maxx,maxy,ss,dX);%,h1,'b'); 
else
  [t_h,t_nout]=statespace3Dhist(minx,miny,minz,maxx,maxy,maxz,ss,tX);%,h1,'r');
  [d_h,d_nout]=statespace3Dhist(minx,miny,minz,maxx,maxy,maxz,ss,dX);%,h1,'b');  
end


%probability in 3D in org data
K=numel(t_h);
alpha=1e-6;
K=numel(t_h);

ind1=sum(sum(sum(t_h)));
p_t_h=(t_h+alpha)./(ind1+alpha*K);     %Laplace smoothing

ind2=sum(sum(sum(d_h)));
outliers=ind1-ind2;
p_d_h=(d_h+alpha)./(ind2+alpha*K);      %Laplace smoothing

%KL divergence in time (integrate over space)
KL=sum(sum(sum( p_t_h.*(log(p_t_h)-log(p_d_h))))); %KL(p(x)|p(x|z))
KL2=sum(sum(sum( p_d_h.*(log(p_d_h)-log(p_t_h))))); %KL(p(x|z)|p(x))
disp(['KL=' num2str(KL)])

%GAUSS:
if size(red,2)>1
   [d_hGauss,d_noutG]=statespace2Dhist(minx,miny,maxx,maxy,ss,dXG);%,h1,'b'); 
else
  [d_hGauss,d_noutG]=statespace3Dhist(minx,miny,minz,maxx,maxy,maxz,ss,dXG);%,h1,'b');
end


%probability in 3D in org data
K=numel(t_h);
alpha=1e-6;
K=numel(t_h);
% 
% ind1=sum(sum(sum(t_hGauss)));
% p_t_h=(t_hGauss+alpha)./(ind1+alpha*K);     %Laplace smoothing

ind2G=sum(sum(sum(d_hGauss)));
outliersG=ind1-ind2G;
p_d_hGauss=(d_hGauss+alpha)./(ind2G+alpha*K);      %Laplace smoothing

%KL divergence in time (integrate over space)
KLGauss=sum(sum(sum( p_t_h.*(log(p_t_h)-log(p_d_hGauss))))); %KL(p(x)|p(x|z))
KLGauss2=sum(sum(sum( p_d_hGauss.*(log(p_d_hGauss)-log(p_t_h))))); %KL(p(x|z)|p(x))
disp(['KLGauss=' num2str(KLGauss)])
%----------------------------------------------------------------------
