%% Simulation of Lorenz Data and according Class Labels:
%Based on gk_get_NoisyLorentTraj
clear all
close all;
clc
path(path,pwd);
dim = 3; 
if dim==1
    d = 2;
    r=0;%input('choose categorization procedure');
    if d==1
        pato=[pwd,'/LorenzData/Data/3Dclassification/RedLorenz/Xclass/'];
        mkdir(pato)
    elseif d==2
        pato=[pwd,'/LorenzData/Data/3Dclassification/RedLorenz/Yclass/'];
        mkdir(pato)
    else
        pato=[pwd,'/LorenzData/Data/3Dclassification/RedLorenz/Zclass/'];
        mkdir(pato)
    end
else
    pato=[pwd, '/LorenzData/Data/3Dclassification/NoisyLorenz/']; 
end

%% System Simulation:
r=28;
s=10;
b=8/3;  % chaos

Ntraj=1;
tvec=0:0.02:40-0.02;


% traj. from set of diff. initial conditions
opt=odeset('RelTol',1e-5,'AbsTol',1e-8);
rand('state',0); randn('state',0);

vmin=[-20 -30 0];
vmax=[20 30 50];
s2=(vmax-vmin)./100;
s1=0.001;
sObs=0.1;

for j=1:101
    if j<10
        num=['0' num2str(j)];
    else 
        num=num2str(j);
    end
    str=['lorenz_traj_chaos_n' num2str(Ntraj) '_T' num2str(length(tvec)-1000) '_' num2str(s1) '_' num '_',num2str(sObs),'.mat'];

    
    T=cell(1,Ntraj);
    Z=zeros(3,length(tvec)-1000,Ntraj);
    
    option=1; %1=Runge-Kutta, 2=matlab ODE solver
    for i=1:Ntraj
        
        switch option
            case 1    % Runge-Kutta (adds noise directly)
                v0=(vmax-vmin).*rand(1,3)+vmin;
                [t,v]=RK_Lorenz(tvec,v0,r,s,b,s1); v=v';%si=noise level
               
                v=v(1001:end,:);
                T{i}=t';
                E=randn(size(v,1),3).*(ones(size(v,1),1)*sObs);
                
                Znoise(:,:,i)=v';
                Znoisetrans(:,:,i)=ztransfo(Znoise(:,:,i))+E';
                Z(:,:,i)=v';%+E';
                Ztrans(:,:,i)=ztransfo(Z(:,:,i));
                noise=s2;

                
            case 2    % ODE solver
                v0=(vmax-vmin).*rand(1,3)+vmin;
                [t,v]=ode23(@LorenzEqns,tvec,v0,opt,r,s,b);        
                v=v(1001:end,:);
                
                % add obs noise manually
                E=randn(size(v,1),3).*(ones(size(v,1),1)*s2);
                T{i}=t';
                Znoise(:,:,i)=v';
                Znoisetrans(:,:,i)=ztransfo(Znoise(:,:,i))+E'; %observation noise
                Z(:,:,i)=v';
                Ztrans(:,:,i)=ztransfo(Z(:,:,i));
                
                noise=s2;
        end
    end
    X=Z;
    Xtrans=Ztrans;
    Xnoise=Znoise;
    Xnoisetrans=Znoisetrans;
    resolution=T;
    T=length(tvec);
    switch dim
        case 1
            try 
                C=Classification_3dim(X);
                Ctrans=Classification_3dim(Xtrans);
                Cnoise=Classification_3dim(Xnoise);
                Cnoisetrans=Classification_3dim(Xnoisetrans);
            catch exc
                disp(exc)
            end
        case 3
            try 
                C=Classification_3dim(X);
                Ctrans=Classification_3dim(Xtrans);
                Cnoise=Classification_3dim(Xnoise);
                Cnoisetrans=Classification_3dim(Xnoisetrans);
            catch exc
                disp(exc)
            end   
    end
    save([pato str],'X','Xtrans','Xnoise','Xnoisetrans','C','Ctrans','Cnoise','Cnoisetrans','T','resolution','r','s','b','noise');
    
        
end

%% Function for basic z-Transformation:
function Ztrans=ztransfo(Z)

    Ztrans=zeros(size(Z,1),size(Z,2));

    
    for j=1:size(Z,1)
        muZ=mean(Z(j,:)); %mean per state
        varZ=std(Z(j,:),1);
        Ztrans(j,:)=(Z(j,:)-muZ)/varZ;
    end 
    
    end
    
