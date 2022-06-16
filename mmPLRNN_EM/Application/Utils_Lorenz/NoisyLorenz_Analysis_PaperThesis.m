%% Lorenz System PLRNN Analysis PAPER/THESIS:
function NoisyLorenz_Analysis_PaperThesis(config)
% 30.09.19
% Philine Bommer
    close all
    clc
    % 
    
    pat = config.pat;
    snr = config.snr;
    resultpat = config.patR;
    pat2 = config.patnet;
    tlen = config.tlen;
    sysnr = [];
    
    patEV = resultpat;

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

    for RUN=1:snr
        try
        pato=pat2;
        filename=['NoisyLorenz_statistics_FullEM_sysnr_' num2str(RUN) '.mat'];

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

        Ntraj=1;
        catch exc
            disp('not converged')
        end
    end

    

    %% Part 1: Comparison prediction Gauss-only vs. MC extension (from z_drawn)

    nst = size(sysnr,1);
    [n,T]=size(Xinit{nst});
    X_init=cell(1,nst);
    notconv=0;
    
    Corr_MC_pred= zeros(n,nst);
    Corr_Gauss_pred= zeros(n,nst);

    Diff_MC_pred= zeros(n,nst);
    Diff_Gauss_pred= zeros(n,nst);

    Corr_MC_10= zeros(n,nst);
    Corr_Gauss_10= zeros(n,nst);

    Diff_MC_10= zeros(n,nst);
    Diff_Gauss_10= zeros(n,nst);

    XPred=cell(2,nst);
    X_Pred10=cell(2,nst);
    X_init=cell(1,nst);

    notconv=0;
    %Generate x for inferred Z:
    for nr=1:nst
        %Set Variables of according sys:

        if ~isempty(Mu0Gauss{nr,1})
        nrs=sysnr(nr);
        if size(Ez{nr},1)>1

            s2=config.s2;
            sObs=0.1;
            if nrs<10
                str=['lorenz_traj_chaos_n' num2str(Ntraj) '_T' num2str(tlen) '_' num2str(s2) '_0' num2str(nrs) '_' num2str(sObs) '.mat'];
            else
                str=['lorenz_traj_chaos_n' num2str(Ntraj) '_T' num2str(tlen) '_' num2str(s2) '_' num2str(nrs) '_' num2str(sObs) '.mat'];
            end

            filename2=strcat(pato,str);
            syst=load(filename2);
            X=syst.Xtrans;
            X_init{nr}=X;
        end
       

        [n,T]=size(X);
        Ts=10*T;
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
        %Generation of obs.:
        for t=1:Ts
            X_pred(:,t)=BEst{nr}{1}*max(Zdrawn(:,t),0);
            X_predG(:,t)=BGauss{nr}{1}*max(ZdrawnGauss(:,t),0);
        end
        for t=1:T
            X_pred10(:,t)= BEst{nr}{1}*max(Zahead(:,t),0);
            X_pred10G(:,t)= BGauss{nr}{1}*max(ZaheadG(:,t),0);
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
    red=[];

    for nr=1:nst

        if ~isempty(Mu0Gauss{nr,1})
        %Prepare necessary network parameters:
        dat=struct();
        dat.AEst=AEst{nr};
        dat.WEst=WEst{nr};
        dat.CEst=CEst{nr};
        dat.hEst=hEst{nr};
        dat.BEst=BEst{nr};
        dat.mu0Est=Mu0Est{nr}{1,1};
        dat.AGauss=AGauss{nr};
        dat.WGauss=WGauss{nr};
        dat.CGauss=CGauss{nr};
        dat.hGauss=hGauss{nr};
        dat.BGauss=BGauss{nr};
        dat.mu0Gauss=Mu0Gauss{nr}{1,1};
        v0=X_init{nr}(:,1);

        [KL,KLGauss,KL2,KL2Gauss]=add_KLDiv_01(dat,v0,red);
        KL2_div(nr,1)=KL2;
        KL2_div(nr,2)=KL2Gauss;
        KL_div(nr,1)=KL;
        KL_div(nr,2)=KLGauss;
        clear dat
        else
           notconv=notconv+1;
           KL_div(nr,:)=nan;
           KL2_div(nr,:)=nan;
        end
    end

    KL_div(isnan(KL_div(:,1)),:)=[];
    KL2_div(isnan(KL2_div(:,1)),:)=[];


    %% relative KL
    KL_rel=KL_div(:,1)/max(max(KL_div));
    KL_rel=[KL_rel,(KL_div(:,2)/max(max(KL_div)))]; %correction by absolute max. approx.


    fileEV='NoisyLorenz_Evaluation.mat';
    save([patEV,fileEV],'X_init','Xinit','XPred','XPred10','KL_rel')

end

