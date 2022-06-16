%% Testing M-step:
% Philine Bommer 20.03.19
% INPUT:
% 
%
% OUTPUT:
% Param: struct containing all model Parameter estimates

function [ Param ] = Testing_Mstep_MultiCat(CtrPar,obj,Sd,NR,U,Ez)

%     tol=CtrPar(1);
%     MaxIter=CtrPar(2);
%     tol2=CtrPar(3);
%     eps=CtrPar(4);
%     flipOnIt=CtrPar(5);
%     FinOpt=CtrPar(6);   % quad. prog. step at end of E-iterations
    fixedS=CtrPar.fixedS;   % S to be considered fixed or to be estimated
    fixedC=CtrPar.fixedC;   % C to be considered fixed or to be estimated
    fixedB=CtrPar.fixedB;   % B to be considered fixed or to be estimated
    fixedG=CtrPar.fixedG;   % G to be considered fixed or to be estimated
    h_est=obj.h;
    T=obj.T;
    Inp=obj.Inp;
    X=Sd.x;
    C_=Sd.c;
    
    mu0_est=obj.mu0;
    W_est=obj.W;
    A_est=obj.A;
    S_est=obj.Sigma;
    C_est=obj.C;
    h_est=obj.h;
    
    XZsplit=[];
    B0=[]; 
    G0=[]; 
    
    
%     nbatch=length(X);   % number of batches
    m=length(h_est);
    EV= [];   
    
    %calculates every necessary unit from passed variables:
    [Ephizi,Ephizij,Eziphizj,Vest]=ExpValPLRNN2(Ez,U);
    issymmetric(Vest)
    %calculates necessary sums:
    EV=UpdateExpSums(Ez,Vest,Ephizi,Ephizij,Eziphizj,X,Inp,EV);
    
    EV.cat=Sd.c;
    Beta=[];
    
    EV.Ezi=Ez;
    EV.Ez=Ez;
    [B_est,G_est,Beta_est,G_theta]=ParEstPLRNNobsNS(EV,XZsplit,B0,G0,T,Vest,Beta,Sd.c,NR);
%     if ChunkList(k)<=nbatch, N=size(X{nb+1},1); end
    N=size(X,1);
    EV.E1p=zeros(m); EV.F1=zeros(N,m); EV.F2=zeros(N,N);
        
         % M-step process model -----------------------------------
%     if fixedS, S0=S_est; else S0=[]; end
%     if fixedC, C0=C_est; else C0=[]; end
    S0=[];
    C0=[];
    lam=0;
    Lb=[]; Ub=[];
    [mu0_est,W_est,A_est,S_est,C_est,h_est]=ParEstPLRNNlat(EV,S0,C0,Lb,Ub,lam);
%     %AUSKOMMENTIERT für Lorenz Auswertung
     
    %Output Parameter estimates:
    Param.B=B_est;
    Param.G=G_est;
    if NR.use
        Param.Beta=Beta_est;
        Param.G_theta=G_theta;
    end
    Param.mu0=mu0_est;
    Param.W=W_est;
    Param.A=A_est;
    Param.S=S_est;
    Param.C=C_est;
    Param.h=h_est;
    Param.V=Vest;
    Param.Ephizi=Ephizi;
    Param.Ephizij=Ephizij;
    %Finish Param stuff
    

end