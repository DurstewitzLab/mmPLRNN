function [t,x]=RK_Lorenz(tvec,x0,r,s,b,si)
%LorenzEqns(t,v,r,s,b)
%4th order RK for Lorenz path integration

x=zeros(length(x0),length(tvec));
x(:,1)=x0;
for t=2:length(tvec)
    eps=si*randn(3,1);
    dt=tvec(t)-tvec(t-1);
    k1=LorenzEqns(0,x(:,t-1),r,s,b)+eps/dt;
    k2=LorenzEqns(dt/2,x(:,t-1)+dt/2*k1,r,s,b)+eps/(2*dt);
    k3=LorenzEqns(dt/2,x(:,t-1)+dt/2*k2,r,s,b)+eps/(2*dt);
    k4=LorenzEqns(dt,x(:,t-1)+dt*k3,r,s,b)+eps/dt;
    x(:,t)=x(:,t-1)+dt/6*(k1+2*k2+2*k3+k4);
end;

    