function [ U_full,U ] = CalculateHessian(obj,Sd,CtrPar)
%% Function performing the Hessian calculation:
% For testing purposes, since we do not need a full E-step for the
% M-step testing
% INPUT: obj - class containing all necessary model parameter 
%        Sd - class containg all simulated states
%        CtrPar - struct containing control parameter

    % DEFINE PARAMETERS:
    S=obj.Sigma;
    Inp_ = obj.Inp;
    mu0_=obj.mu0;
    G=obj.Gamma;
    C_=Sd.c;
    X_=Sd.x;
    Ez=Sd.z;
    A=obj.A;
    W=obj.W;
    C=obj.C;
    B=obj.B;
    h=obj.h;
    Beta=obj.Beta;
    m=obj.p;
    Tsum=obj.T;
    
    %Perform part of the E-step:-------------------------------------------
    [ U0,U1,U2 ] = ConstructMatrices(obj, Sd, CtrPar);
    
    %Calculation of U and D:-----------------------------------------------
     d0=zeros(1,m*Tsum); d0(Ez>0)=1; D0=spdiags(d0',0,m*Tsum,m*Tsum);
     H=D0*U1; U=U0+D0*U2*D0+H+H';
        
    %Calculate non-simplifiable derivations:
    Ez=reshape(Ez,m*Tsum,1);
        [C_ddot] = C_deriv(Beta, Ez, Tsum, m, C_);
        %[beta_star_dot] = beta_deriv(Beta, Ez, Sd.c, Tsum, m);
    
    U_full = -U - C_ddot;

end

function [beta_star_dot] = beta_deriv(Bet, z, c, T, m)
%Function forming the beta_star vector and its derivative(according to eq.23,29) 

%IN CASE OF SPARSE DATA: beta_star and beta_star_dot will have entries at
%missing time step t which are zero

    beta_star = zeros(T,1);
    beta_star_dot = zeros(T*m,1);
    [K,T]=size(c);
    b_zt = zeros(K,1);
    b_ztdot = zeros(K,m);
    
    for t=1:T
        z_t=z(((t-1)*m+1):t*m);
        c_t=c(:,t);
        % Calculation of beta_star:
        b_zt(1:end-1)= (z_t'*Bet)';
        b_zt(end)=0;
        beta_star(t)=sqrt(c_t'*b_zt);
        
        % Calculation of first derivative
        b_ztdot(1:end-1,:)=Bet';
        b_ztdot(end,:)=zeros(1,m);
        beta_star_dot(((t-1)*m+1):t*m)=(c_t'*b_ztdot)';
         
    end

end


function [C_ddot] = C_deriv(Bet, z, T, m, c)
%Function of derivative corresponding to MC dist.
% C is a vector of size Tx1
% C_dot is a vector of size MTx1
%     C = zeros(T,1);
%     C_dot = [];
%     gamma_ztDot = zeros(m,1);
%     %calculate C & C_ddot entry by entry:
%     for i=1:T
%         c_t=c(:,i);
%         if sum(c_t)==0
%         %IN CASE of sparse data zeros at missing timestep:
%            C(i)=0;
%            C_dot=[C_dot;zeros(m,1)];
%         else
%             z_t = z(((i-1)*m+1):i*m);
%             temp=sum(exp(z_t'*Bet));
%             gamma_zt = sqrt(log(1+temp));
%             C(i)=gamma_zt;
%             for j=1:m
%                 temp2 = Bet(j,:)*exp(z_t'*Bet)';
%                 gamma_ztDot(j)=temp2/(1+temp);
%             end
%             C_dot=[C_dot;gamma_ztDot];
%         end
%     end
    
    Gammas_ddot = cell(T);
    gamma_ddot = zeros(m,m);
    for i=1:T
        c_t=c(:,i);
        if sum(c_t)==0
        %IN CASE of sparse data zeros at missing timestep:
            Gammas_ddot{i}=zeros(m,m);
        else
            z_t = z(((i-1)*m+1):i*m);
            temp = 1+sum(exp(z_t'*Bet));
            denom= temp^2;
            for l=1:m
                temp2 = Bet(l,:)*exp(z_t'*Bet)';
                for k=1:m
                    temp3 = sum(Bet(k,:)*exp(z_t'*Bet)');
                    temp4 = sum(Bet(l,:).*Bet(k,:).*exp(z_t'*Bet));
                
                    gamma_ddot(k,l) =(temp*temp4 - temp3*temp2)/denom;
                end  
            end
            Gammas_ddot{i}= gamma_ddot;
        end
    end

    C_ddot=blkdiag(Gammas_ddot{:});
end


function [ U0,U1,U2 ] = ConstructMatrices(obj, Sd,CtrPar)
    % DEFINE PARAMETERS:
    S=obj.Sigma;
    Inp_ = obj.Inp;
    mu0_=obj.mu0;
    G=obj.Gamma;
    C_=Sd.c;
    X_=Sd.x;
    Ez=Sd.z;
    A=obj.A;
    W=obj.W;
    C=obj.C;
    B=obj.B;
    h=obj.h;
%     m=obj.p;
%     Tsum=obj.T;
    %beta_star=obj.beta;
    Beta=obj.Beta;
    CtrPar.d0=[];
    tol=CtrPar.tol;
    flipAll = CtrPar.flipAll;
    eps = CtrPar.eps;
    FinOpt = CtrPar.FinOpt;
    
    m=length(A);    % # of latent states

    if iscell(X_), X=X_; Inp=Inp_; mu0=mu0_;
    else X{1}=X_; Inp{1}=Inp_; mu0{1}=mu0_; end;
    ntr=length(X);  % # of distinct trials

    if iscell(C_), c=C_;
    else c{1}=C_;
    end
    [K,J] = size(C_);
    
%%% construct block-banded components of Hessian U0, U1, U2, and 
% vectors/ matrices v0, v1, V2, V3, V4, as specified in the objective 
% function Q(Z), eq. 7, in Durstewitz (2017)
u0=S^-1+A'*S^-1*A; K0=-A'*S^-1;
Ginv=G^-1;
u2A=W'*S^-1*W; u2B=B'*Ginv*B; u2=u2A+u2B;
u1=W'*S^-1*A; K2=-W'*S^-1;
U0=[]; U2=[]; U1=[];
v0=[]; v1=[];
Tsum=0;

    
for i=1:ntr   % acknowledge temporal breaks between trials
    T=size(X{i},2); Tsum=Tsum+T;
    U0_ = repBlkDiag(u0,T);
    KK0 = repBlkDiag(K0,T);

    X0=X{i};
    tm=find(sum(isnan(X0)));
    
    if isempty(tm)
        U2_ = repBlkDiag(u2,T);
    else
        %U2_ = BlkDiagU2(u2A,u2B,X{i},B,Ginv);  % if just individ. elements xit are to be removed
        U2_ = BlkDiagU2(u2A,u2B,T,tm);  % removes whole time points tm with missing values
    end
    
    U1_ = repBlkDiag(u1,T);
    KK2 = repBlkDiag(K2,T);
    
    KK0=blkdiag(KK0,K0);
    kk=(T-1)*m+1:T*m; U0_(kk,kk)=S^-1;
    U0_=U0_+KK0(m+1:end,1:T*m);
    KK0=KK0'; U0_=U0_+KK0(1:T*m,m+1:end);
    U2_(kk,kk)=B'*G^-1*B;
    U1_(kk,kk)=0; KK2=blkdiag(KK2,K2);
    U1_=U1_+KK2(m+1:end,1:T*m);
    U0=sparse(blkdiag(U0,U0_)); U2=sparse(blkdiag(U2,U2_)); U1=sparse(blkdiag(U1,U1_));
    
    I=C*Inp{i}+repmat(h,1,T);
    vka=S^-1*I; vka(:,1)=vka(:,1)+S^-1*(mu0{i}-h); vkb=A'*S^-1*I(:,2:T);
    v0_=(vka(1:end)-[vkb(1:end) zeros(1,m)])'; v0=[v0;v0_];
    
    X0(:,tm)=0; % zero out time points with missing values completely;
    % to zero out only individ. components, for each component xit=nan the
    % i-th row of B has to be set to 0, ie corresp. columns of vka need to
    % be computed such that all rows of B and xt corresp. to missing val.
    % are =0.
    vka=B'*G^-1*X0;
    vkb=-W'*S^-1*I(:,2:T);
    v1_=(vka(1:end)+[vkb(1:end) zeros(1,m)])'; v1=[v1;v1_];
end;


% %categorical extension:
% %Actually the only part which makes C_ necessary...
% 
% if isempty(Beta)
%     Beta = rand(m,K); %init guess
% end
% if isempty(beta_star)
%         beta_star = zeros(m,T);
%         for r=1:T  
%          beta_star(:,r)=Beta*C_(:,r);
%         end   
%         beta_star = reshape(beta_star,m*T,1);
% else
%     [a,b]=size(beta_star);
%     if b>1
%         beta_star=reshape(beta_star,m*T,1);
%     end
%  end
% clear a b
end

%% HELPER:
function [ BigM ] = repBlkDiag( M, number_rep )
% repeats the Matrix M NUMBER_REP times in the block diagonal

MCell = repmat({M}, 1, number_rep);
BigM = blkdiag(MCell{:});

end


%     % block-diag matrix with missing observations
%     % --- excluding **individual components** {xit}:
%     function U2m=BlkDiagU2(u2A,u2B,X0,B,L)
%         U2m=[];
%         for t=1:size(X0,2)
%             s=find(~isnan(X0(:,t)));
%             if length(s)==size(X0,1), u2x=u2B;
%             elseif length(s)==0, u2x=0;
%             else u2x=B(s,:)'*L(s,s)*B(s,:); end 
%             U2m=blkdiag(U2m,u2A+u2x);
%         end
%     end

    % block-diag matrix with missing observations
    % --- excluding only whole time points in tm
    function U2m=BlkDiagU2(u2A,u2B,T,tm)
        U2m=[];
        for t=1:T
            if ismember(t,tm), U2m=blkdiag(U2m,u2A);
            else U2m=blkdiag(U2m,u2A+u2B); end
        end
    end

