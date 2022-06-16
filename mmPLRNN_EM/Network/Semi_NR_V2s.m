%% Semi Newton-Rapshon Version 2(UPDATE 01.05.19):



function [ z,n,LogLike,Z,U_bar ] = Semi_NR_V2s(Input, zvs, options)
%Function performing  Semi-NR algorithm

    Usv=Input.Usv;
    U0=Input.U0;
    U1=Input.U1;
    U2=Input.U2;
    v0=Input.v0;
    v1=Input.v1;
    c=Input.C_;
    if iscell(c) c=cell2mat(c); end;
    Bet=Input.Beta;
    dsv=Input.dsv;
    T=Input.T;
    Tsum=Input.Tsum;
    m=Input.m;
    y=Input.y;
    k=Input.k;
    idx=Input.idx;
    
    %Define options:
    maxIterations=80;
    g=options.g_min;
    bound=options.bound;
    flipAll=options.flipAll;
    tol=options.tol;
    dErr=options.dErr;
    
    LogLike = [];
    Z=[];

    %init values for the step NR-step:
    totDev = 1000; 
    TotDev = 1000; 
    z = zvs;
    U = Usv;
    d = dsv;
    v = (v0+d'.*v1);
    Z=z;

  
    [C, C_dot, C_ddot] = C_deriv(Bet, z, T, m, c);
    [beta_star,beta_star_dot] = beta_deriv(Bet, z, c, T, m);
    
    %Initialization for the loglikelihood:
     d0=zeros(1,m*Tsum); d0(z>0)=1; D0=spdiags(d0',0,m*Tsum,m*Tsum);
     H=D0*U1; U=U0+D0*U2*D0+H+H';
     vv=v0+d0'.*v1;
     loglike =-1/2*(z'*U*z-z'*vv-vv'*z +2*(C'*C)-2*(beta_star'*beta_star));
     LLH=[loglike-10^(-2),loglike];
     
     n=1;
     Err=1e16;
     flipAll=true;
     q=2;
     numviolatedCon=[];
     numflip=1000;
     U_bar=-U;
    
while LLH(q)>=LLH(q-1) && isempty(k) && ~isempty(idx) %&& abs(TotDev)>bound && q<3000%&& numflip>0.1 %&& dErr<tol*Err(q)
        
    clear LL w
    Zsv=z; Usv=full(U_bar); dsv=d; %save last step
    LL=[];
    w=1;
    
    %Recalculation of U and D
        D=spdiags(d',0,m*Tsum,m*Tsum);
        H=D*U1; U=U0+D*U2*D+H+H';     

        
    %Convergence possibilities:
         % track log-likelihood 
         vv=v0+d'.*v1;
         loglike =-1/2*(z'*U*z-z'*vv-vv'*z+2*(C'*C)-2*(beta_star'*beta_star));
         LogLike = [LogLike,loglike];
         
         
    
    while abs(totDev)>bound && w<maxIterations
        

         % save last step
        zsv=z; 

        
        grad_z = -C_dot - U*zsv + v + beta_star_dot;
        grad_z = sparse(grad_z);
        U_bar = -U - C_ddot;
        U_bar = sparse(U_bar);
        %NR-step:
        z = zsv -U_bar\(g*grad_z);
        
        %Calculate non-simplifiable derivations:
        [C, C_dot, C_ddot] = C_deriv(Bet, z, T, m, c);
        [beta_star,beta_star_dot] = beta_deriv(Bet, z, c, T, m);
        
        %Recalculation of U and D
        D=spdiags(d',0,m*Tsum,m*Tsum);
        H=D*U1; U=U0+D*U2*D+H+H';
        n=n+1;
        
        
        %Convergence possibilities:
         % track log-likelihood (if desired)
         vv=v0+d'.*v1;
         loglike =-1/2*(z'*U*z-z'*vv-vv'*z +2*(C'*C)-2*(beta_star'*beta_star));
         LogLike = [LogLike,loglike];
         LL= [LL,loglike];
          
         if w>1
            [totDev] = Bound_Calc(LL,loglike);
         end
         disp([num2str(w) '    ' num2str(totDev)])
         
         w=w+1;
         
    end
    if w==maxIterations
        disp('max Iter met!!!')
        disp(w)
        disp(totDev)
    end
    LLH = [LLH,loglike];

    
    
    %flip violated constraint(s):
        idx=find(abs(d-(z>0)')); %finds all violated constraints
        ae=abs(z(idx));
        n=n+1;%int8(n+1);
        q=q+1;
        disp(n)

        if flipAll, d(idx)=1-d(idx);    % flip all constraints at once
        else [~,r]=max(ae); d(idx(r))=1-d(idx(r)); end; % flip constraints only one-by-one, 
                
        numviolatedCon=[numviolatedCon,length(idx)];
        
        %Calculate Convergence Criterion(enable criterion in while loop if necessary):
        [TotDev] = Bound_Calc(LLH,LLH(end));
    
         
end
   
   if LLH(q)<LLH(q-1) 
      %the overall LogLike drops again choose step before as opt:
      z=Zsv;
      U_bar=full(Usv);
      d=dsv;
   end

             disp([num2str(length(idx)) '   ' num2str(length(k)) '    ' num2str(TotDev)])

end


function [beta_star,beta_star_dot] = beta_deriv(Bet, z, c, T, m)


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


function [C, C_dot, C_ddot,Gammas_ddot ] = C_deriv(Bet, z, T, m, c)
%Function of derivative corresponding to MC dist.

    C = zeros(T,1);
    C_dot = [];
    gamma_ztDot = zeros(m,1);
    %calculate C & C_ddot entry by entry:
    for i=1:T
        c_t=c(:,i);
        if sum(c_t)==0
        %IN CASE of sparse data zeros at missing timestep:
           C(i)=0;
           C_dot=[C_dot;zeros(m,1)];
        else
            z_t = z(((i-1)*m+1):i*m);
            temp=sum(exp(z_t'*Bet));
            gamma_zt = sqrt(log(1+temp));
            C(i)=gamma_zt;
            for j=1:m
                temp2 = Bet(j,:)*exp(z_t'*Bet)';
                gamma_ztDot(j)=temp2/(1+temp);
            end
            C_dot=[C_dot;gamma_ztDot];
        end
    end
    
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

function [totDev] = Bound_Calc(LL,ll)
%Function to calculate the deviations in order to look for
%convergence
    if length(LL)>=3
        totDev= -mean(LL(end-2:end) -ll);
    else 
        totDev= -mean(LL -ll);
    end

end