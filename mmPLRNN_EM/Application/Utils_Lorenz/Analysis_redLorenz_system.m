%% Evaluation of the Lorenz system with reduced dimensions:
function Analysis_redLorenz_system(pat,patdat,patResults,snr)
clc

runP=1:snr;

pato = patdat;
patEV = patResults;
    Ez=[];
    Mu0Est=[];
    BEst=[];
    GammaEst=[];
    AEst=[];
    WEst=[];
    CEst=[];
    hEst=[];
    SigmaEst=[];
    BetaEst=[];
    
    EzGauss=[];
    BetaGauss=[];
    BGauss=[];
    CGauss=[];
    GammGauss=[];
    SigmaGauss=[];
    hGauss=[];
    Mu0Gauss=[];
    AGauss=[];
    WGauss=[];
    Xinit=[];
    
    LLAll=[];
    LLAllGauss=[];
    sysnr=[];
    red=1;
    if red
        d = 4;
        red=[red,d];
        if d==4
            r_d=0;
        end
    end

for RUN=1:length(runP)
    try

    filename = ['ReducedDimLorenzsystem_3dim_statistics_FullEM_sysnr_',num2str(RUN), '.mat'];
    
    syst=load([pat filename]);
    nr=1;
    sysnr=[sysnr,RUN];
    
    % Gauss:
    ezGauss{nr}=syst.Gauss.Ezi;
    betaGauss{nr}=syst.Gauss.Beta;
    bGauss{nr}=syst.Gauss.B;
    cGauss{nr}=syst.Gauss.C;
    gammGauss{nr}=syst.Gauss.G;
    sigmaGauss{nr}=syst.Gauss.S;
    HGauss{nr}=syst.Gauss.h;
    mu0Gauss{nr}=syst.Gauss.mu0;
    aGauss{nr}=syst.Gauss.A;
    wGauss{nr}=syst.Gauss.W;
    lLAllGauss{nr}=syst.Gauss.LL;
    % MC:
    ez{nr}=syst.MM.Ezi;
    mu0Est{nr}=syst.MM.mu0;
    bEst{nr}=syst.MM.B;
    gammaEst{nr}=syst.MM.G;
    aEst{nr}=syst.MM.A;
    wEst{nr}=syst.MM.W;
    cEst{nr}=syst.MM.C;
    HEst{nr}=syst.MM.h;
    sigmaEst{nr}=syst.MM.S;
    betaEst{nr}=syst.MM.Beta;
    lLAll{nr}=syst.MM.LL;
    % Initial:
    xinit{nr}=syst.Init.Xinit;
    Xred{nr}=syst.Init.X;
    
    
    
    Ez=[Ez;ez];
    Mu0Est=[Mu0Est;mu0Est];
    BEst=[BEst;bEst];
    GammaEst=[GammaEst;gammaEst];
    AEst=[AEst;aEst];
    WEst=[WEst;wEst];
    CEst=[CEst;cEst];
    hEst=[hEst;HEst];
    SigmaEst=[SigmaEst;sigmaEst];
    BetaEst=[BetaEst;betaEst];
    
    EzGauss=[EzGauss;ezGauss];
    BetaGauss=[BetaGauss;betaGauss];
    BGauss=[BGauss;bGauss];
    CGauss=[CGauss;cGauss];
    GammGauss=[GammGauss;gammGauss];
    SigmaGauss=[SigmaGauss;sigmaGauss];
    hGauss=[hGauss;HGauss];
    Mu0Gauss=[Mu0Gauss;mu0Gauss];
    AGauss=[AGauss;aGauss];
    WGauss=[WGauss;wGauss];
    
    Xinit=[Xinit;xinit];
    
    LLAll=[LLAll;lLAll];
    LLAllGauss=[LLAllGauss;lLAllGauss];
    
    Ntraj=syst.Ntraj;
    T=syst.T;
    catch exc
        disp('not converged')
    end
    
end

nst=size(Mu0Est,1);


%% Part 1: Reapply M-step
red=1;
    PARM=cell(nst,1);
    PARM_G=cell(nst,1);
%Predefined Parameters:
   Cstr.Lb=[]; Cstr.Ub=[]; Cstr.XZspl=[]; Cstr.lam=0;
    
    NR=struct;
    NR.g=0.001;
    NR.bound=0.01;
    NR.use=true;
    NR.fixedBeta=0;
    
   % Define CtrPar/NR struct:
   CtrPar=struct;
   CtrPar.tol = 1*10^(-2);
   CtrPar.MaxIter = 51; %Max iterations EM algo
   CtrPar.tol2 = 0;
   CtrPar.eps = 1*10^(-5);
   CtrPar.flipOnIt = 10;
   CtrPar.exp=false;
   CtrPar.boundE=0.01;
   CtrPar.gE=1;
   CtrPar.maxItr=6;
   CtrPar.outbnd=10;% Outerbound Estep
   CtrPar.FinOpt = 0;   % quad. prog. step at end of E-iterations
   CtrPar.fixedS = false;   % S to be considered fixed or to be estimated
   CtrPar.fixedC = true;   % C to be considered fixed or to be estimated
   CtrPar.fixedB = false;   % B to be considered fixed or to be estimated
   CtrPar.fixedG= false;
   CtrPar.d0=[];
   CtrPar.flipAll = false;
   CtrPar.eps = 1*10^(-5);
   CtrPar.FinOpt = 0;

notconv=0;
Ntraj=1;

for nr=1:nst
    
    
    nrs=sysnr(nr);
    
    if length(Ez{nr})>1
    if ~red
        s2=0.001;
        sObs=0.001;
        if nrs<10
            str=['lorenz_traj_chaos_n' num2str(Ntraj) '_T' num2str(1000) '_' num2str(s2) '_0' num2str(nrs) '_' num2str(sObs) '.mat'];
        else
            str=['lorenz_traj_chaos_n' num2str(Ntraj) '_T' num2str(1000) '_' num2str(s2) '_' num2str(nrs) '_' num2str(sObs) '.mat'];
        end
%     filename2=string(strcat(pato,str));
        filename2=strcat(pato,str);
        syst=load(filename2);
        X=syst.Xtrans;
        X_init{nr}=X;
        C_=syst.Ctrans;
        C_init{nr}=C_;
    else
        s2=0.001;
        sObs=0.001;
        if nrs<10
            str=['lorenz_traj_chaos_n' num2str(Ntraj) '_T' num2str(1000) '_' num2str(s2) '_0' num2str(nrs) '_' num2str(sObs) '.mat'];
        else
            str=['lorenz_traj_chaos_n' num2str(Ntraj) '_T' num2str(1000) '_' num2str(s2) '_' num2str(nrs) '_' num2str(sObs) '.mat'];
        end
        filename2=strcat(pato,str);
        syst=load(filename2);
        X_init{nr}=syst.Xtrans;
        X=X_init{nr};
        C_=syst.Ctrans;
        C_init{nr}=C_;
    end

    %Set Variables of according sys:
    mu0=Mu0Est{nr,1};
    m=size(mu0{1,1},1);
    [n,Ts]=size(X);
    try
    z= reshape(Ez{nr},m,Ts);
    catch exc
        disp(exc)
    end
    zGauss= reshape(EzGauss{nr},m,Ts);
    q=20;
    a=-1; b=1;
    Obs.z=z;
    Obs.x=X;
    Obs.c=C_;
    Inpar.Sigma=SigmaEst{nr};
    Inpar.Inp=zeros(q,Ts);
    Inpar.mu0=mu0{1,1};
    Inpar.Gamma=diag(diag(s2*rand(n)));
    Inpar.A=AEst{nr};
    Inpar.W=WEst{nr};
    Inpar.C=CEst{nr};
    Inpar.B=a+(b-a).*rand(n,m);
    Inpar.h=hEst{nr};
    Inpar.Beta=BetaEst{nr}{1,1};
    Inpar.p=m;
    Inpar.T=Ts;
    
    ObsG.x=X;
    ObsG.z=zGauss;
    ObsG.c=C_;
    InparG.Sigma=SigmaGauss{nr};
    InparG.Inp=zeros(q,Ts);
    InparG.mu0=Mu0Gauss{nr,1}{1,1};
    InparG.Gamma=diag(diag(s2*rand(n)));
    InparG.A=AGauss{nr};
    InparG.W=WGauss{nr};
    InparG.C=CGauss{nr};
    InparG.B=a+(b-a).*rand(n,m);
    InparG.h=hGauss{nr};
    InparG.Beta=BetaEst{nr}{1,1};
    InparG.p=m;
    InparG.T=Ts;
%     try

    [ U_full,~ ] = CalculateHessian(Inpar,Obs,CtrPar);

    [ ~,U ] = CalculateHessian(InparG,Obs,CtrPar);
    
    %Generation of obs.:
    CtrPar.fixedS=false;   % S to be considered fixed or to be estimated
    CtrPar.fixedC=false;   % C to be considered fixed or to be estimated
    CtrPar.fixedB=false;   % B to be considered fixed or to be estimated
    CtrPar.fixedG=false;
    U_full=-U_full;
    
    [ Param ] = Testing_Mstep_MultiCat(CtrPar,Inpar,Obs,NR,U_full,z);
    PARM{nr}=Param;
    
    NR.use=false;
    
    [ Param_Gauss ] = Testing_Mstep_MultiCat(CtrPar,InparG,Obs,NR,U,zGauss);
    PARM_G{nr}=Param_Gauss;
%     catch exc
%         disp(exc)
%         notconv=notconv+1;
%         PARM{nr}=0;
%         PARM_G{nr}=0;
%     end 
    else %if not converged
        notconv=notconv+1;
        PARM{nr}=0;
        PARM_G{nr}=0;
         
    end
end

%% Part 1B: Comparison prediction Gauss-only vs. MC extension (from z_drawn)

XPred=cell(2,nst);
X_Pred10=cell(2,nst);
X_init=cell(1,nst);

notconv=0;
%Generate x for inferred Z:
for nr=1:nst
    %Set Variables of according sys:
    if isstruct(PARM{nr}) 
        
     if ~red
        filename2=strcat(pato,str);
        syst=load(filename2);
        X=syst.Xtrans;
        X_init{nr}=X;
     else
        filename2=strcat(pato,str);
        syst=load(filename2);
        X_init{nr}=syst.Xtrans;
        X=X_init{nr};
     end
    ParamObs=PARM{nr};
    ParamBObs=PARM_G{nr};
    
    [n,T]=size(X);
    Ts=100*T;
    X_pred= zeros(n,Ts);
    X_predG= zeros(n,Ts);
    X_pred10= zeros(n,T);
    X_pred10G= zeros(n,T);
    
    %Drawing Z(full pred. + 10-step-ahead pred.)
    mu0=Mu0Est{nr}{1,1};
    m=size(mu0,1);
    mu0Gauss=Mu0Gauss{nr}{1,1};
    A=AEst{nr};
    AG=AGauss{nr};
    W=WEst{nr};
    WG= WGauss{nr};
    h= hEst{nr};
    hG= hGauss{nr};
    
    z= reshape(Ez{nr},m,T);
    zGauss= reshape(EzGauss{nr},m,T);
    
    Zdrawn=zeros(m,Ts);
    ZdrawnGauss=zeros(m,Ts);
    
    Zdrawn(:,1)=mu0;
    ZdrawnGauss(:,1)=mu0Gauss;
    
    Time=1*Ts;
    
    %Full timeline prediction:
    for t=2:Ts
        Zdrawn(:,t)=A*Zdrawn(:,t-1)+W*max(Zdrawn(:,t-1),0)+h;
        ZdrawnGauss(:,t)=AG*ZdrawnGauss(:,t-1)+WG*max(ZdrawnGauss(:,t-1),0)+hG;
    end
    
    %10 steps ahead pred.:
    steps=10;
    Zahead= zeros(m,T);
    ZaheadG= zeros(m,T);
    for ts=1:T-9
        Zt10=zeros(m,10);
        Zt10Gauss=zeros(m,10);
        if ts==1
            Zt10(:,1)= mu0;
            Zt10Gauss(:,1)= mu0Gauss;
        else
            Zt10(:,1)= z(:,ts);
            Zt10Gauss(:,1)= zGauss(:,ts);
        end
        for t=2:steps
            Zt10(:,t)= A*Zt10(:,t-1)+W*max(Zt10(:,t-1),0)+h;
            Zt10Gauss(:,t)= AG*Zt10Gauss(:,t-1)+WG*max(Zt10Gauss(:,t-1),0)+hG;
        end
        if ts==1
            Zahead(:,1:10)=Zt10;
            ZaheadG(:,1:10)= Zt10Gauss;
        else
            Zahead(:,ts+9)= Zt10(:,10);
            ZaheadG(:,ts+9)= Zt10Gauss(:,10);
        end
        
    end
    
    B=ParamObs.B;
    BG=ParamBObs.B;
    
    %Generation of obs.:
    for t=1:Ts
        X_pred(:,t)=B*max(Zdrawn(:,t),0);
        X_predG(:,t)=BG*max(ZdrawnGauss(:,t),0);
    end
    for t=1:T
        X_pred10(:,t)= B*max(Zahead(:,t),0);
        X_pred10G(:,t)= BG*max(ZaheadG(:,t),0);
    end
    

    XPred{1,nr}=X_pred;
    XPred{2,nr}=X_predG;
    
    XPred10{1,nr}=X_pred10;
    XPred10{2,nr}=X_pred10G;
    
    else
        notconv=notconv+1;
        XPred{1,nr}=[];
        XPred{2,nr}=[];
    
        XPred10{1,nr}=[];
        XPred10{2,nr}=[];
    end
end



%% Part 2: KL divergence
KL_div=zeros(nst,2);
KL2_div=zeros(nst,2);
notconv=0;

for nr=1:nst
    
    if isstruct(PARM{nr})
    %Prepare necessary network parameters:
    ParamObs=PARM{nr};
    ParamBObs=PARM{nr};
    dat=struct();
    dat.AEst=AEst{nr};
    dat.WEst=WEst{nr};
    dat.CEst=CEst{nr};
    dat.hEst=hEst{nr};
    BE{1}= ParamObs.B;
    dat.BEst=BE;
    dat.mu0Est=Mu0Est{nr}{1,1};
    dat.AGauss=AGauss{nr};
    dat.WGauss=WGauss{nr};
    dat.CGauss=CGauss{nr};
    dat.hGauss=hGauss{nr};
    BEG{1}=ParamBObs.B;
    dat.BGauss=BEG;
    dat.mu0Gauss=Mu0Gauss{nr}{1,1};
    v0=X_init{nr}(:,1);
    red=1;
    
    [KL,KLGauss,KL2,KL2Gauss,d_noutG,d_nout]=add_KLDiv_01(dat,v0,red);
    if d_nout<20000
    KL2_div(nr,1)=KL2;
    KL_div(nr,1)=KL;
    else 
    KL2_div(nr,1)=nan;
    KL_div(nr,1)=nan;    
    end
    if d_noutG<20000
    KL2_div(nr,2)=KL2Gauss;
    KL_div(nr,2)=KLGauss;
    else
    KL2_div(nr,2)=nan;
    KL_div(nr,2)=nan;    
    end
    clear dat
    else
       notconv=notconv+1;
       KL_div(nr,:)=nan;
       KL2_div(nr,:)=nan;
    end
end
[min_num,min_idx] = min(KL_div(:,1));

KL_div(isnan(KL_div(:,1)),:)=[];
KL2_div(isnan(KL2_div(:,1)),:)=[];

%% Relative KLx
KL_rel=KL_div(:,1)/max(max(KL_div));
KL_rel=[KL_rel,(KL_div(:,2)/max(max(KL_div)))];

files = ['RedLorenz_Evaluation_3dim_RedY_', num2str(nst),'_sys.mat'];

save([patEV,files],'X_init','Xinit','XPred','XPred10','KL_rel')

end



