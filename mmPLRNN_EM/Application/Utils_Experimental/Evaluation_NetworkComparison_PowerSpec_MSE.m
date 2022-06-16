%% Evaluation of the comparitive training PLRNN vs mmPLRNN:
function Evaluation_NetworkComparison_PowerSpec_MSE(config)

clc
close all
%% Step 1: Choose dataset according to training settings and initiation
pat = config.pat;
% Load Data from according path:
type = config.type; %specify data folder/typ used for the training
set = config.sett; %Name of preprocessed dataset
fnOut{1}=[pat '/Data/Training/' type set]; 
d = load(fnOut{1});
Data=d.Data;

if strcmp(config.patnum,'full')
    patnum = length(Data);
else 
    patnum = config.patnum;
end
    

% Set Folder for Trained Networks of according Dataset:
fnL{1}=[pat '/Data/Evaluation/NetComparison/' type];
i=0;

%Specify training conditions:
FULL= config.Full; 
NoInp = config.noinp;
M=config.numlat;

% Set Sub-Folders for Trained Networks according to training conditions:
fnL{1}=[pat '/Data/Evaluation/NetComparison/' type];
if FULL
    fnL{2}=[fnL{1} '/FullTS'];
    if NoInp
        fnL{3}=[fnL{2} '/NoInp'];
    else
        fnL{3}=[fnL{2} '/Inp'];
    end
else
    fnL{2}=[fnL{1} '/TrialWise'];
    if NoInp
        fnL{3}=[fnL{2} '/NoInp'];
    else
        fnL{3}=[fnL{2} '/Inp'];
    end
end
fnL{4} = [fnL{3} '/m' num2str(M) '/'];

% Free data generation:
XPred10=cell(1,1);
XPred10G=cell(1,1);
Xtrue=cell(1,1);
ZinfG=cell(1,1);
Zinf=cell(1,1);
ZAhead=cell(1,1);
ZAhead_G=cell(1,1);
%MSE calculation
nahead = config.nahead;
%Best init cond.:
LL_G=zeros(26,1);
LLM=zeros(26,1);
m=M;
j=0;

%% Evaluation Loop:
for steps=1:size(nahead,2)
    
    
for pt=1:patnum
    %Pre-allocation of arrays:
    LL_mm=[];
    LL_g=[];
    LLm=zeros(5,2);
    LLg=zeros(5,2);
    
    
    for sr=1:5
        try
%% Load Data and set up eval structure:
        i=i+1;
        if steps==1
            j=j+1;
        end



        fnOut{4}=[fnL{4} 'mmPLRNN_m' num2str(m) 'pat_' num2str(pt) '_init_' num2str(sr) '.mat'];
        fnOut{5}=[fnL{4} 'Sparse_PLRNN_m' num2str(m) 'pat_' num2str(pt) '_init_' num2str(sr) '.mat'];

        mmSys=load(fnOut{4});
        GaussSys=load(fnOut{5});
        if steps==1
            SysPar{1,j}=mmSys;
            SysPar{2,j}=GaussSys;
        end

        GaussSys.B=GaussSys.BG;
        GaussSys.C=GaussSys.CG;
        GaussSys.A=GaussSys.AG;
        GaussSys.W=GaussSys.WG;
        GaussSys.h=GaussSys.hG;
        GaussSys.mu0=GaussSys.mu0G;
        GaussSys.S=GaussSys.SG;
        GaussSys.G=GaussSys.GG;
        GaussSys.Ezi=GaussSys.EziG;

%% Prediction of firing rate (gaussian data) for both networks:
        Z=mmSys.Ezi;
        if iscell(Z)
            Z=Z{1,1};
        end
        Zinf{1,i}=Z;
        T=size(Z,2);
        M=size(Z,1);
        Inp=mmSys.Inp;
        X=mmSys.Xn{1,1};
        mu0_=mmSys.mu0{1,1};
        mu0_G=GaussSys.mu0{1,1};

        ZG=GaussSys.Ezi;
        if iscell(ZG)
            ZG=ZG{1,1};
        end
        ZinfG{1,i}=ZG;
        XG=GaussSys.Xn{1,1};
        N=size(XG,1);
        B_=mmSys.B;
        B_g=GaussSys.BG;
        B=cell2mat(B_);
        BG=cell2mat(B_g);


        Ts=T;
        Zdrawn=zeros(m,Ts);    
        Zdrawn(:,1)=mmSys.mu0{1,1};
        ZdrawnG=zeros(m,Ts);    
        ZdrawnG(:,1)=GaussSys.mu0{1,1};
        A=mmSys.A;
        W=mmSys.W;
        h=mmSys.h;
%         C=mmSys.C;
        AG=GaussSys.AG;
        %BG=GaussSys.BG;
        WG=GaussSys.WG;
        hG=GaussSys.hG;
%         CG=GaussSys.CG;

        %10-step ahead prediction:
        sts=nahead(steps);
        Zahead= zeros(M,T);
        ZaheadG= zeros(M,T);
        for ts=1:T-(sts-1)
            Zt10=zeros(m,sts);
            Zt10G=zeros(m,sts);
            if ts==1
                Zt10(:,1)= mu0_;
                Zt10G(:,1)= mu0_G;
            else
                Zt10(:,1)= Z(:,ts);
                Zt10G(:,1)= ZG(:,ts);
            end
            for t=2:sts
                Zt10(:,t)= A*Zt10(:,t-1)+W*max(Zt10(:,t-1),0)+h;
                Zt10G(:,t)= AG*Zt10G(:,t-1)+WG*max(Zt10G(:,t-1),0)+hG;
            end
            if ts==1
                Zahead(:,1:sts)=Zt10;
                ZaheadG(:,1:sts)=Zt10G;
            else
                Zahead(:,ts+(sts-1))= Zt10(:,sts);
                ZaheadG(:,ts+(sts-1))= Zt10G(:,sts);
            end
        end
        ZAhead{1,i}=Zahead;
        ZAhead_G{1,i}=ZaheadG;
        X_pred10 = zeros(N,T);
        X_pred10G = zeros(N,T);
        for t=1:T
            X_pred10(:,t)= B*max(Zahead(:,t),0);
            X_pred10G(:,t)= BG*max(ZaheadG(:,t),0);
        end
        XPred10{1,i}=X_pred10;
        XPred10G{1,i}=X_pred10G;

        

%% Full timeline prediction:
        for t=2:Ts
            Zdrawn(:,t)=A*Zdrawn(:,t-1)+W*max(Zdrawn(:,t-1),0)+h;%+C*Inp(:,t);
            ZdrawnG(:,t)=AG*ZdrawnG(:,t-1)+WG*max(ZdrawnG(:,t-1),0)+hG;%+CG*InpG(:,t);
        end 
        ZPred{1,i}=Zdrawn;
        ZPredG{1,i}=ZdrawnG;

        %Generation of obs.:
        X_est=zeros(N,Ts);
        X_estG=zeros(N,Ts);
        for t=1:Ts
            X_est(:,t)=B*max(Z(:,t),0);%+diag(G{1});
            X_estG(:,t)=BG*max(ZG(:,t),0);
        end
        if steps==1
        XEst{1,j}=X_est;
        XEstG{1,j}=X_estG;
        end


        X_free=zeros(N,Ts);
        X_freeG=zeros(N,Ts);
        for t=1:Ts
            X_free(:,t)=B*max(Zdrawn(:,t),0);%+diag(G{1});
            X_freeG(:,t)=BG*max(ZdrawnG(:,t),0);
        end
        if steps==1
        XFree{1,j}=X_free;
        XFreeG{1,j}=X_freeG;
        Xtrue{1,j}=X;
        end

%% Safes Index of best. Init-cond. run:
        LL_mm =[LL_mm,max(mmSys.LL)];
        LL_g=[LL_g,max(GaussSys.LLG)];

        if steps==1
            LLm(sr,1)=max(mmSys.LL);
            LLm(sr,2)=sr;
            LLg(sr,1)=max(GaussSys.LLG);
            LLg(sr,2)=sr;
        end    
        catch exc
            disp(exc)
            disp('Did not converge')
        end
    end
 
end
end

%% SAVE: 
% Saves all created arrays of the evaluation according to the evaluated
% network training
fnL{5}=[pat '/Data/Evaluated/NetComparison/' type];
mkdir(fnL{5})
if FULL
    fnL{6}=[fnL{5} '/FullTS'];
    mkdir(fnL{6})
    if NoInp
        fnL{7}=[fnL{6} '/NoInp'];
    else
        fnL{7}=[fnL{6} '/Inp'];
    end
else
    fnL{6}=[fnL{5} '/TrialWise'];
    mkdir(fnL{6})
    if NoInp
        fnL{7}=[fnL{6} '/NoInp'];
    else
        fnL{7}=[fnL{6} '/Inp'];
    end
end

mkdir(fnL{7});
fnL{8} = [fnL{7} '/m' num2str(M) '/'];
mkdir(fnL{8});

fnOut{6}=[fnL{8} '/Eval_PLRNN_mmPLRNN_m' num2str(m) '.mat'];

save(fnOut{6});

%% SAVES: important arrays for paper plot
out=[fnL{8} '/Paper_Sparse_PLRNN_mmPLRNN_m' num2str(m) '.mat'];

save(out,'nahead','XPred10','XPred10G','X','XG','XFree','XFreeG');
