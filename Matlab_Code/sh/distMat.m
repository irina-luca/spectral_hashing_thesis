function D=distMat(P1, P2)
%
% Euclidian distances between vectors
% Each vector is one row
  
% disp("# -- distMat called -- #");


if nargin == 2
%     disp("# -- distMat: nargin == 2 -- #")
    P1 = double(P1);
    P2 = double(P2);
%     printVar(P1, "P1");
%     printVar(P2, "P2");
    X1=repmat(sum(P1.^2,2),[1 size(P2,1)]);
%     printVar(X1, "X1");
    X2=repmat(sum(P2.^2,2),[1 size(P1,1)]);
%     printVar(P2.^2, "P2.^2");
%     printVar(sum(P2.^2,2), "sum(P2.^2,2)");
%     printVar(X2, "X2");
    R=P1*P2';
%     printVar(R, "R");
    D=real(sqrt(X1+X2'-2*R));
%     printVar(D, "D")
else
%     disp("# -- distMat: [nargin !== 2] -- #")
    P1 = double(P1);
%     printVar(sum(P1.^2,2), "sum(P1.^2,2)")
    % NOTE(*): Repeat each entry in vector sum(P1.^2,2) in 
    % dimensions [1 size(previous matrix)].
    X1=repmat(sum(P1.^2,2),[1 size(P1,1)]); 
%     printVar(X1, "X1")
    % NOTE(*): P1 x transpose(P1); R is symmetric (spectral theorem), 
    % maybe it's trying to make P1 diagonalizable?!, not sure.
    R=P1*P1';
%     printVar(R, "R")
    D=X1+X1'-2*R;
    D = real(sqrt(D));
%     printVar(D, "D")s
end


