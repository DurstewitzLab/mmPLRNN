%% Newton-Rapshon(M-step):
%Using the Newton-Raphson approach(according to Semi-NR)


%OUTPUT:
%Beta: MxK matrix containg beta vectors of each category 


function [ Beta,G_theta ] = Full_NR(NR,Cs,Beta,EV)
%Function performing  Full-NR algorithm 
    
    g_b=NR.g;
    bound=NR.bound;
    
    Grad_beta=0;
    J=0;
   
    count=0;
    G_theta = [];
    Criterion=1000;
    
    if iscell(EV.Varz)
        GT=0;
        if isempty(Beta), Beta=rand(M,K-1); end
        [K,~]=size(Cs{1});
        [M,~]=size(EV.Ez{1});
        for bat=1:length(EV.Varz)
            C_=Cs{bat};
            V=EV.Varz{bat};

            [~,T]=size(EV.Ez{bat});
            E=reshape(EV.Ez{bat},M*T,1);
            [K,~]=size(C_);
    
            clear t
    
            if isempty(Beta), Beta=rand(M,K-1); end
    
            Grad_beta=Grad_beta+GradientBeta(E,V,Beta,T,C_);
            J=J+HessianBeta(E,V,Beta,T,C_);
            [gt]=CalcuGtheta(E,V,Beta,T,C_);
            GT=GT+gt;
            
        end
        G_theta=GT;
    else
        if iscell(Cs), C_=cell2mat(Cs);
        else C_=Cs; end
        if iscell(EV.Varz), V=cell2mat(EV.Varz);
        else V=EV.Varz;
        end
        if iscell(EV.Ez), EV.Ez=cell2mat(EV.Ez); end
        [M,t]=size(EV.Ez);
        Ez=reshape(EV.Ez,M*t,1);
        E=Ez;
        if EV.T==t
           T=EV.T;
        else
           T=t;
        end
          [K,t]=size(C_);
    
        clear t
    
        if isempty(Beta), Beta=rand(M,K-1); end
    
        Grad_beta=GradientBeta(E,V,Beta,T,C_);
        J=HessianBeta(E,V,Beta,T,C_);
        [G_theta]=CalcuGtheta(E,V,Beta,T,C_);
    end
     
    BETAS = Beta;
    GRAD = Grad_beta;
      while Criterion>=bound
    
        count=count+1;
    
        Beta_old=reshape(Beta,M*(K-1),1);
        BETAS=[BETAS,Beta];
        
    %full NR step:-----------------------------------

        Beta = Beta_old - J\(g_b*Grad_beta);
    
    %update Gradient and Hessian:--------------------
        Beta=reshape(Beta,M,(K-1));
        
        if iscell(EV.Varz)
            GT=0;
            for bat=1:length(EV.Varz)
                V=EV.Varz{bat};
                C_=Cs{bat};

                [~,T]=size(EV.Ez{bat});
                E=reshape(EV.Ez{bat},M*T,1);
   
                Grad_beta=Grad_beta+GradientBeta(E,V,Beta,T,C_);
                J=J+HessianBeta(E,V,Beta,T,C_);
                [gt]=CalcuGtheta(E,V,Beta,T,C_);
                GT=GT+gt;
            
            end
            G_theta=[G_theta,GT];
        else
   
            Grad_beta=GradientBeta(E,V,Beta,T,C_);
            J=HessianBeta(E,V,Beta,T,C_);
            [g_theta]=CalcuGtheta(E,V,Beta,T,C_);
            GRAD=[GRAD,Grad_beta];
            G_theta=[G_theta,g_theta];
            
        end

    %Convergence criterion:--------------------------
        [Criterion]=ConvergenceCriterion(count,G_theta);
        currentstep=count
        Crit=Criterion
      end

    
end

function[ Grad_beta ]=GradientBeta(E,V,Beta,T,C_)

[M,K]=size(Beta);
MT=length(E);
M=MT/T;

Ts=sum(sum(C_)); % counts time steps including categorical information
grad_beta=zeros(M,K,Ts);
ts=0;

%Calculation of Jacobian matrix:---------------------------------------------
for i=1:T
    c_t=C_(:,i);
    
    if sum(c_t)==0 %if sparse do not include
    else %time steps including cat. data
    ts=ts+1;
    %%Start Calculations:--------------------------------------------------
    E_t=E(((i-1)*M+1):i*M);
    V_t=V(((i-1)*M+1):i*M,((i-1)*M+1):i*M);
    Beta_t=Beta;
   
    %Denominator(const for each derivative):-----------------------
    temp=0;
    for j=1:K
            beta_j=Beta_t(:,j);
            temp=temp+exp(beta_j'*E_t +((beta_j'*V_t)*beta_j)/2);
    end
    denom=1+temp;
    position=find(c_t==1);
    if position<K+1
        
        for l=1:K
            beta_l=Beta_t(:,l);
        
            nominator=(E_t+V_t*beta_l)*exp(beta_l'*E_t +((beta_l'*V_t)*beta_l)/2);
            rest=nominator/denom;
        
            %two cases:-------------------------------------
            if l==position
            grad_beta(:,l,ts)=(E_t-rest);
            else  
            grad_beta(:,l,ts)=-rest;
            end
        end   
    else
    %Last category was chosen:
          for l=1:K
            beta_l=Beta_t(:,l);
        
            nominator=(E_t+V_t*beta_l)*exp(beta_l'*E_t +((beta_l'*V_t)*beta_l)/2);
            rest=nominator/denom;
          
            grad_beta(:,l,ts)=-rest;
            
          end
    end
    end
end
Grad_beta=sum(grad_beta,3);
Grad_beta=reshape(Grad_beta,M*K,1);

end

function[ J_beta ]=HessianBeta(E,V,Beta,T,C_)
MT=length(E);
M=MT/T;
[K,T]=size(C_);
Ts=sum(sum(C_)); % counts time steps including categorical information
J_beta=zeros(M*(K-1),M*(K-1),Ts);
ts=0;

for i=1:T
    c_t=C_(:,i);
    
    if sum(c_t)==0 %
    else %
    ts=ts+1;
    
    %%Start Calculations:--------------------------------------------------
    E_t=E(((i-1)*M+1):i*M);
    V_t=V(((i-1)*M+1):i*M,((i-1)*M+1):i*M);
    Beta_t=Beta;

    %calculate denominator:-----------------------------------
        temp=0;
        for j=1:K-1
            beta_j=Beta_t(:,j);
            temp=temp+exp(beta_j'*E_t +((beta_j'*V_t)*beta_j)/2);
        end
        denom=1+temp;
    
        for l=1:K-1
            beta_l=Beta_t(:,l);
            exp_bl=exp(beta_l'*E_t-((beta_l'*V_t)*beta_l)/2);
        
            for m=1:K-1
                beta_m=Beta_t(:,m);
                exp_bm=exp(beta_m'*E_t-((beta_m'*V_t)*beta_m)/2);
            
                mat_prod=((E_t-V_t*beta_l)*(E_t-V_t*beta_m)');
                %2 cases(diagonal and off-diagonal):-------------
                if l==m
                    part1=(mat_prod*exp_bl)/denom;
                    part2=(mat_prod*exp_bl*exp_bl)/denom^2;
                    part3=(V_t*exp_bl)/denom;
                    J_beta(((l-1)*M+1):l*M,((m-1)*M+1):m*M,ts)= -part1+part2-part3;    
                else
                    nom1=mat_prod*exp_bl*exp_bm;
                    J_beta(((l-1)*M+1):l*M,((m-1)*M+1):m*M,ts)= -nom1/(denom^2); 
                end
            
            end
        end
    

    end
end

  %sum over time:--------------------------------------------
  J_beta=sum(J_beta,3);


end

function[ g_theta ] =CalcuGtheta(E,V,Beta,T,C_)

    
    MT=length(E);
    m=MT/T;
    [k,T]=size(C_);
    Ts=sum(sum(C_)); 
    g_t=zeros(Ts,1);
    ts=0;
    

    
    
    for t=1:T
        c_t=C_(:,t);
    
        if sum(c_t)==0 
        else 
            ts=ts+1;
    
        %%Start Calculations:----------------------------------------------
        
        %Parameter Preperation:
            E_t=E(((t-1)*m+1):t*m);
            V_t=V(((t-1)*m+1):t*m,((t-1)*m+1):t*m);

            position=find(c_t==1);
            sum_exp=0;
            for j=1:k-1
                beta_j=Beta(:,j);
                sum_exp=sum_exp+exp(beta_j'*E_t +((beta_j'*V_t)*beta_j)/2);
            end
            if position<k
                beta_star=Beta(:,position);
                g_t(t)=(beta_star'*E_t)-log(1+sum_exp);
            else
                g_t(t)=-log(1+sum_exp);
            end
        end
    end
    g_theta=sum(g_t,1);
    
end

function Criterion = ConvergenceCriterion(count,G_theta)
    if count==0
        Criterion=100;
    elseif count<=6
        disp('g_t')
        disp(G_theta(count))
        Criterion=abs(mean(G_theta-G_theta(count)));
    else
        Criterion=abs(mean(G_theta((count-6):(count-1))-G_theta(count)));
    end
end
