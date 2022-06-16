function [x,z]=runsys(A,W,h,C,mu0,Inp,B,T,Sigma,Gamma,Beta)

% simulate system with final parameters
[N,M]=size(B);
% [M,K]= size(Beta);
z=zeros(M,T);
% K=K+1;
if nargin < 9
    err=zeros(M,T);
else
    err=mvnrnd(zeros(1,M), Sigma,T)';
end
if nargin < 10
    errobs=zeros(M,T);
else
    errobs=mvnrnd(zeros(1,N), Gamma,T)';
end

z(:,1)=mu0+C*Inp(:,1);
for t=2:T
    z(:,t)=A*z(:,t-1)+W*max(z(:,t-1),0)+h+C*Inp(:,t);%+err(:,t);
end

x=zeros(N,T);
for t=1:T
    x(:,t)=B*max(z(:,t),0);%+errobs(:,t);
end




end