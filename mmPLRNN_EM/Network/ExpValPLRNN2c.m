function [Ephizi,Ephizij,Eziphizj,V]=ExpValPLRNN2c(Ez,U)
%
% please cite (& consult for further details): Durstewitz (2017) PLoS Comp Biol
% 
% computes expectancies E[phi(z)], E[z phi(z)'], E[phi(z) phi(z)'],  
% as given in eqn. 10-15, based on provided state expectancies and Hessian
% 
% REQUIRED INPUTS:
% Ez: MxT matrix of state expectancies as returned by StateEstPLRNN
% U: negative Hessian of log-likelihood returned by StateEstPLRNN 
% h: Mx1 vector of thresholds
%
% OUTPUTS:
% Ephizi: E[phi(z)]
% Ephizij: E[phi(z) phi(z)']
% Eziphizj: E[z phi(z)']
% V: state covariance matrix E[zz']-E[z]E[z]'

eps=1e-4;   % minimum enforced variance/ eigenvalue

[m,T]=size(Ez);

%% invert block-tridiagonal neg. Hessian U
U0=zeros(m*T,2*m);
for t=1:T
    k0=(t-1)*m+1:t*m; %t for z_t 
    k1=max(0,(t-2)*m)+1:t*m; %t for z_t-1
    U0(k0,1:2*m)=[zeros(m,(t<2)*m) U(k0,k1)]; 
end;
V0=invblocktridiag(U0,m); %Covariance matrix from hessian(e-step)
v=V0(m+1:end,1:m)';
for t=1:T-1, V0((t-1)*m+1:t*m,2*m+1:3*m)=v(1:m,(t-1)*m+1:t*m); end;
k0=(1:T*m)'*ones(1,3*m); %k0 dim T*m x 3*m
k1=ones(T*m,1)*(1:3*m);
k1(2*m+1:end,:)=k1(2*m+1:end,:)+reshape(repmat(m:m:(T-2)*m,m,3*m),(T-2)*m,3*m);
k1((T-1)*m+1:T*m,:)=k1((T-2)*m+1:(T-1)*m,:);
V0(1:m,:)=circshift(V0(1:m,:),-m,2);
V0((T-1)*m+1:T*m,:)=circshift(V0((T-1)*m+1:T*m,:),m,2);
V=sparse(k0,k1,V0);

% ensure proper covariance matrix
V=(V+V')./2;    % ensure symmetry
for i=1:m*T, V(i,i)=max(V(i,i),eps); end;   % restrict min var
%ensure pos. definite covariance matrix
eigenV=eig(V);
if min(eigenV)<=0
    Vold=V;
    EPS=repmat(eps,T*m);
    if size(diag(EPS),2)==1
        V=V+EPS;
    else
        V=V+diag(EPS);
    end
    if min(eig(V))<=0
        V=nearestSPDc(full(Vold)); % Find closest pos. def. Matrix (min. Frobenius Form)
    end
end



Ez=Ez(1:end)';
v=spdiags(V,0);
s=sqrt(v);
fk=normpdf(zeros(size(Ez)),Ez,s+zeros(size(Ez)));
Fk=1-normcdf(zeros(size(Ez)),Ez,s+zeros(size(Ez)));


%% E[phi(zi)]
vfk=v.*fk;%sigma_k^2*N_k
Ephizi=vfk+Ez.*Fk;


%% E[phi(zi)*phi(zj)]

Ephizij=sparse(m*T,m*T);
EzzV=sparse(m*T,m*T);
o1=ones(m*2,1);
for t=1:T-1
    k0=(t-1)*m+1:(t+1)*m;
    %k1=t*m+1:(t+2)*m;
    lam_l=(o1*v(k0)')./(v(k0)*v(k0)'-V(k0,k0).*V(k0,k0));
    lam_l_inv=1./lam_l;
    mu_kl=o1*Ez(k0)'-V(k0,k0).*((Ez(k0)./v(k0))*o1');
    Flk=1-normcdf(zeros(2*m),mu_kl',sqrt(lam_l_inv));
    Fkl=1-normcdf(zeros(2*m),o1*Ez(k0)',sqrt(lam_l_inv)');
    Nlk=normpdf(zeros(2*m),mu_kl',sqrt(lam_l_inv));
    EzzV(k0,k0)=Ez(k0)*Ez(k0)'+V(k0,k0);%(z_k^max*z_l^max+sigma_kl^2)
    Ephizij(k0,k0)=lam_l_inv'.*(o1*fk(k0)').*(Nlk./lam_l+mu_kl'.*Flk)+ ...
        (v(k0)*Ez(k0)'.*(fk(k0)*o1')+EzzV(k0,k0).*(Fk(k0)*o1')).*Fkl; %equation 15 lam_inv=Lam_k?
    %if t<T-1, Ephizij(k1,k1)=0; end;
end;
Ephizij=(Ephizij+Ephizij')./2;%mean?


%% E[zi*phi(zj)]
Eziphizj=sparse(m*T,m*T);
for t=1:T-1
    k0=(t-1)*m+1:(t+1)*m;
    Eziphizj(k0,k0)=EzzV(k0,k0).*(o1*Fk(k0)')+Ez(k0)*vfk(k0)';
end;


%% E[phi(zi)*phi(zi)] = E[zi*phi(zi)] for h=0!!!

Ephizij=triu(Ephizij,1)+tril(Ephizij,-1)+diag(diag(Eziphizj,0));


%%
% (c) 2017 Daniel Durstewitz, Dept. Theoretical Neuroscience, Central
% Institute of Mental Health, Heidelberg University
