%% a function that computes the block diagonal and block-off-diagonal
%% elements of a symmetric block tridiagonal matrix' inverse

% INPUT:
% A: symmetric block tridiagonal matrix with blocks of size m x m
% m: optional, size of block, when m = 1, not block diagonal anymore
% A is input in two block columns of size n x m each; n is the number of blocks
% in A; i.e., full A has a size of (n x m)x(n x m) 
% first block column is for the lower off-diagonal elements; the first block in it is all
% zeros, since the off diagonal is 1 block shorter than diagonal
% second block column is the diagonal elements; no need to compute the full
% inverse for the current applications. Making the function more general is
% for future work
% OUTPUT:
% Ainv: the block elements at the diagonal and lower off-diagonal elements
% of A's inverse.


function Ainv = invblocktridiag(A,m)

if nargin < 1
  error('not enough input arguments to INVTRIDIAG');
end

if nargin < 2
  m = 1;
end

nm = size(A,1);
n  = nm/m;
Ainv = zeros(nm,2*m);

% asymmetric matrix
if size(A,2)/m == 3
    error('asymmetric matrix; not yet implemented');

% symmetric matrix
elseif size(A,2)/m == 2
    cL = zeros(nm,m);
    dL = zeros(nm,m);
    dL(1:m,1:m) = A(1:m,m+1:end);
    for c = 1:n-1
        cL(c*m+1:(c+1)*m,:) = -A(c*m+1:(c+1)*m,1:m)/dL((c-1)*m+1:c*m,:);
        dL(c*m+1:(c+1)*m,:) = A(c*m+1:(c+1)*m,m+1:end) + cL(c*m+1:(c+1)*m,:)*A(c*m+1:(c+1)*m,1:m)';
    end
    
    dR = A(end-m+1:end,m+1:end);
    Ainv(end-m+1:end,m+1:end) = inv(dL(end-m+1:end,:));
    for c = n:-1:2
        Ainv((c-1)*m+1:c*m,1:m) = Ainv((c-1)*m+1:c*m,m+1:end)*cL((c-1)*m+1:c*m,:);
        dR = A((c-2)*m+1:(c-1)*m,m+1:end) - A((c-1)*m+1:c*m,1:m)'*(dR\A((c-1)*m+1:c*m,1:m));
        Ainv((c-2)*m+1:(c-1)*m,m+1:end) = inv(-A((c-2)*m+1:(c-1)*m,m+1:end)+dL((c-2)*m+1:(c-1)*m,:)+dR);
    end
else
    error('input argument has wrong structure');
end


%%
% (c) 2016 Hazem Toutounji, Dept. Theoretical Neuroscience, Central
% Institute of Mental Health, Heidelberg University
