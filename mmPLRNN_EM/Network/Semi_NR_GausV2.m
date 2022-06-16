%% Semi Newton-Rapshon Gauss only data:



function [ z,n,LogLike,Z,U_bar ] = Semi_NR_GausV2(Input, zvs, options)
%Function performing  Semi-NR algorithm

    Usv=Input.Usv;
    U0=Input.U0;
    U1=Input.U1;
    U2=Input.U2;
    v0=Input.v0;
    v1=Input.v1;
    dsv=Input.dsv;
    T=Input.T;
    Tsum=Input.Tsum;
    m=Input.m;
    y=Input.y;
    k=Input.k;
    idx=Input.idx;
    
    %Define options:
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

  
    
    %Initialization for the loglikelihood:
     d0=zeros(1,m*Tsum); d0(z>0)=1; D0=spdiags(d0',0,m*Tsum,m*Tsum);
     H=D0*U1; U=U0+D0*U2*D0+H+H';
     vv=v0+d0'.*v1;
     loglike =-1/2*(z'*U*z-z'*vv-vv'*z);
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
    Zsv=z; Usv=U_bar; dsv=d; %save last step
    LL=[];
    w=1;
    
    %Recalculation of U and D
        D=spdiags(d',0,m*Tsum,m*Tsum);
        H=D*U1; U=U0+D*U2*D+H+H';     

        
    %Convergence possibilities:
         % track log-likelihood 
         vv=v0+d'.*v1;
         loglike =-1/2*(z'*U*z-z'*vv-vv'*z);
         LogLike = [LogLike,loglike];
         

    
    while abs(totDev)>bound 
        

         % save last step
        zsv=z; 
        Z=[Z,z];

        
        grad_z = - U*zvs + v; 
        U_bar = -U; 
        %NR-step:
        z = zvs -U_bar\(g*grad_z);
        
   
        %Recalculation of U and D
        D=spdiags(d',0,m*Tsum,m*Tsum);
        H=D*U1; U=U0+D*U2*D+H+H';
        n=n+1;
        
        
        %Convergence possibilities:
         % track log-likelihood (if desired)
         vv=v0+d'.*v1;
         loglike =-1/2*(z'*U*z-z'*vv-vv'*z);
         LogLike = [LogLike,loglike];
         LL= [LL,loglike];
          
         if w>1
            [totDev] = Bound_Calc(LL,loglike);
         end
         disp([num2str(w) '    ' num2str(totDev)])
         
         w=w+1;
         
    end
    LLH = [LLH,loglike];

    
    
    %flip violated constraint(s):
        idx=find(abs(d-(z>0)')); %finds all violated constraints
        ae=abs(z(idx));
        n=n+1;
        q=q+1;
        disp(n)

        if flipAll, d(idx)=1-d(idx);    % flip all constraints at once
        else [~,r]=max(ae); d(idx(r))=1-d(idx(r)); end; % flip constraints only one-by-one, 
                
        numviolatedCon=[numviolatedCon,length(idx)];
        
        %Calculate Convergence Criterion:
        [TotDev] = Bound_Calc(LLH,LLH(end));
         disp(q)
         disp([num2str(length(idx)) '   ' num2str(length(k)) '    ' num2str(TotDev)])
     
         
end
   
   if LLH(q)<LLH(q-1) 
      z=Zsv;
      U_bar=Usv;
      d=dsv;
   end
    
end




function [totDev] = Bound_Calc(LL,ll)
    if length(LL)>=5
        totDev= abs(mean(LL(end-4:end) -ll));
    else 
        totDev= abs(mean(LL -ll));
    end

end