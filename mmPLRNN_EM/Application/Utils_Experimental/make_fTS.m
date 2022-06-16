function [dat]=make_fTS(dat)
%% Create full TS
% create full time series from trial-based data
% mu0=dat.mu0;
X=dat.Xn;
C_=dat.C_;
S=dat.Inp;



% dat.mu0=cell2mat(mu0);
Xx{1}=cell2mat(X);
Cc{1}=cell2mat(C_);
Inp{1}=cell2mat(S);
dat.Xn=Xx;
dat.C_=Cc;
dat.Inp=Inp;


% for i=1:length(mu0)
    


end