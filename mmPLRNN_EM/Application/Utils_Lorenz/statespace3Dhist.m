function [ histmtx, nout ] = statespace3Dhist(minx,miny,minz,maxx,maxy,maxz,ss,X,hfig,clr,laplace)

%histmtx:   3D matrix with counts in occurrence bins
%nout:      points lying out of range

%GK create count mtx for state space
sx=ss; sy=ss; sz=ss;
x=X(1,:); y=X(2,:); z=X(3,:);

xrange=minx:sx:maxx;
yrange=miny:sy:maxy;
zrange=minz:sz:maxz;

histmtx=zeros(length(xrange)-1,length(yrange)-1,length(zrange)-1);
ix=1;
for i=xrange
    iy=1;
    xbin=[i i+sx];
    for j=yrange
        iz=1;
        ybin=[j j+sy];
        for k=zrange
            zbin=[k k+sz];
            
            ind=sum(x > xbin(1) & x <= xbin(2) & y > ybin(1) & y <= ybin(2) & z > zbin(1) & z <= zbin(2));
            
            histmtx(ix,iy,iz)=ind; %note, 1 too long
            iz=iz+1;
            
        end
        iy=iy+1;
    end
    ix=ix+1;
end
histmtx(ix-1,:,:)=[]; histmtx(:,iy-1,:)=[]; histmtx(:,:,iz-1)=[];

%plot if wanted
if nargin>8
    figure(hfig); hold on; col2=clr;
    nx=size(histmtx,1);
    ny=size(histmtx,2);
    nz=size(histmtx,3);
    for i=1:nx
        for j=1:ny
            for k=1:nz
                ind=histmtx(i,j,k); 
                ms=4;
                if ind~=0, plot3(i,j,k,'s','color',col2,'MarkerSize',ms,'MarkerFaceColor',col2);  end
            end
        end
    end
end


%consistency check
%all outside of grid:
ind=sum(x > xrange(end) | x < xrange(1) | y > yrange(end) | y < yrange(1) | z > zrange(end) | z < zrange(1));
ind2=sum(sum(isnan(X)));
allout=ind + ind2/3;
allin=sum(sum(sum(histmtx)));

txt=sprintf('total val = %2.3f, found val = %2.3f, out of range: %2.3f',length(x), allout+allin, allout);
disp(txt);
nout=allout;







