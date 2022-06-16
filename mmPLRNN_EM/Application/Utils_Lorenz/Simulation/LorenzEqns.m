function dv=LorenzEqns(t,v,r,s,b)
dv(1)=s*(v(2)-v(1));
dv(2)=r*v(1)-v(2)-v(1)*v(3);
dv(3)=v(1)*v(2)-b*v(3);
dv=dv';
