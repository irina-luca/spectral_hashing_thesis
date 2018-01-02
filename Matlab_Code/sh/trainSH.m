function SHparam = trainSH(X, SHparam)
%
% Input
%   X = features matrix [Nsamples, Nfeatures]
%   SHparam.nbits = number of bits (nbits do not need to be 
%   a multiple of 8)
%

disp("# -- trainSH called -- #");

[Nsamples Ndim] = size(X);
% printVar([Nsamples Ndim], "[Nsamples Ndim]");
nbits = SHparam.nbits;

% -- Algorithm -- %
% -- (1) PCA -- %
% printVar(X, "X");
% QUESTION/ THOUGHT: Why take the min here? Don't we always want to have 
% npc=nbits we use to encode? If let's say we have Ndim < nbits, 
% that would be illogical, right? The idea is to dimensionality-reduce, 
% isn't that right?
npca = min(nbits, Ndim); 
printVar(npca, "npca");
printVar(cov(X), "cov(X)");

% NOTE(*): The largest pc along with their eigenvals 
% (in python, signs are weird); From doc: [V,D] = eigs(___) returns 
% diagonal matrix D containing the eigenvalues on the main diagonal, 
% and matrix V whose columns are the corresponding eigenvectors. 
% You can use any of the input argument combinations in previous syntaxes.
% => Then pca contains the largest eigenvectors on each col and l 
% contains the largest eigenvalues on the diagonal.
[pc, l] = eigs(cov(X), npca);
printVar(X, "Xtraining")
printVar(pc, "pc");
printVar(l, "l");
% printVar(X, "X");

% OBS(*): X = X * pc is the transformed dataset after applying the PCA
% transformation with the eigenvectors; so the new X would be the old X
% seen from the new basis/axes
% QUESTION/ THOUGHT: Why did they comment 'no need to remove the mean'?
X = X * pc; 
printVar(X, "X after X = X * pc");
printVar(cov(X), "cov(X transformed)");
% XforTest = X
% disp(XforTest)
% [pcTest, lTest] = eigs(cov(XforTest), npca); 
% printVar(lTest, "lTest");
% OBS(*): After X is PCA-ed, its covariance will have
% the eigenvalues on the diagonal and will also be diagonalized! 
% printVar(cov(X), "cov(X)"); 


% -- (2) Fit uniform distribution -- %
% printVar(eps, "eps");
% printVar(min(X), "min(X)");
% QUESTION/ THOUGHT (1): Why is it redefined?! It actually seems like 
% both approaches give the same result, in case the dataset is very small!!!!
% QUESTION/ THOUGHT (2): I think percentile is better, because it deals 
% with outliers. Or maybe it's the opposite actually.
% NOTE(*): mn and mx defined the second time are each a transposed 
% vector holding the min, respectively max for each dim. of PCA-ed X.
% printVar(mn, "mn after mn = min(X)-eps");
mn = prctile(X, 5); 
% printVar(mn, "prctile(X, 5)");
mn = min(X)-eps;
% printVar(mn, "mn");
mx = prctile(X, 95);
% printVar(mx, "prctile(X, 95)");
mx = max(X)+eps;
% printVar(mx, "mx");
 

% -- (3) Enumerate eigenfunctions -- %
% OBS(*): Maybe it could be smth about how much each pca spans?!??!?!
% OBS(*): R probably contains in each entry the range for 
% the associated dimension.
R=(mx-mn);
% printVar(R, "R = mx - mn");
% OBS(*): max(R) is probably the max range from all the ranges defined
% by the dimensions of the newly transformed matrix X/training set.
% printVar(max(R), "max(R)");
% OBS(*): This is maybe the range for each dimension 
% scaled by (nbits + 1).
% printVar(R/max(R), "R/max(R)");
% printVar((nbits+1)*R/max(R), "(nbits+1)*R/max(R)"); 
% OBS(*): Probably normalized ranges for each 
% dim, but scaled up by nbits+1 and ceiled.
maxMode=ceil((nbits+1)*R/max(R)); 
% printVar(maxMode, "maxMode");
% QUESTION(*): I have no clue how to interpret nModes, but it could be
% the sum of all modes - how many modes there are + 1.
nModes=sum(maxMode)-length(maxMode)+1;
% printVar(nModes, "nModes=sum(maxMode)-length(maxMode)+1;");

% OBS(*): For each pca, we prepare a column of 1's of length that might
% have smth to do with the normalized range for each dim/pca with regards
% to the max range, the # of bits etc.
modes = ones([nModes npca]); 
% printVar(modes, "modes");
m = 1;
for i=1:npca
%     disp("### Inside 1 -> npca loop ###, step " + i);
%     printVar(maxMode(i), "maxMode(i)");
%     printVar(modes(m+1:m+maxMode(i)-1,i), "modes(m+1:m+maxMode(i)-1,i) before");
    modes(m+1:m+maxMode(i)-1,i) = 2:maxMode(i); 
%     printVar(m+maxMode(i)-1, "modes(m+1:m+maxMode(i)-1,i) after");
%     printVar(2:maxMode(i), "Assign/Replace with 2:maxMode(i)");
    % OBS(*): Seems like it updates each piece of the diagonal from
    % 2->whatever number the maxMode(i) decides for the ith pca
%     printVar(modes, "modes");
%     printVar(modes(m+1:m+maxMode(i)-1,i), "modes(m+1:m+maxMode(i)-1,i) after");
    m = m+maxMode(i)-1; 
%     printVar(m, "m");
    % OBS(*): Seems like this gets incremented with the range
    % scaling factor from above?! 
    % OBS(*): Seems like it updates the pcas on diagonal.
    % OBS(*): Probably STUPID, but can it have smth to do with modes
    % of vibration for each dimension?!
end

% printVar(modes, "modes updated after the loop");

modes = modes - 1;

% printVar(modes, "modes after modes = modes - 1");
% printVar(R, "R");

omega0 = pi./R;
% printVar(omega0, "omega0 = pi./R;");
% printVar(repmat(omega0, [nModes 1]), "repmat(omega0, [nModes 1]");
% printVar(modes, "modes");
omegas = modes.*repmat(omega0, [nModes 1]);
printVar(omegas, "omegas = modes.*repmat(omega0, [nModes 1]);");
% NOTE(*): Makes the sum for each row.
eigVal = -sum(omegas.^2,2);
printVar(eigVal, "eigVal = -sum(omegas.^2,2);");
% OBS(*): [B,I] = sort(___) also returns a collection of index vectors 
% for any of the previous syntaxes. I is the same size as A and 
% describes the arrangement of the elements of A into B along the sorted 
% dimension. For example, if A is a vector, then B = A(I). =>
% In our case, we have the following:
        % - yy contains the sorted eigVals
        % - ii contains the position of yy entry in (-eigVal)
[yy,ii]= sort(-eigVal); 
printVar(yy, "yy");
% printVar(ii, "ii");
% printVar(ii(2:nbits+1), "ii(2:nbits+1)");
% printVar(modes, "modes before last step");

% OBS(*): In this case, basically the same as modes([2;3], :) =>
% NO CLUE how to interpret this, but #rows will have to be same as #bits
modes=modes(ii(2:nbits+1),:); 
% printVar(ii(2:nbits+1), "ii(2:nbits+1)");
% printVar(modes, "modes after last step => modes=modes(ii(2:nbits+1),:);");

% 4) store paramaters
SHparam.pc = pc;
SHparam.mn = mn;
SHparam.mx = mx;
SHparam.mx = mx;
SHparam.modes = modes;

% printVar(SHparam.pc, "SHparam.pc");   
% printVar(SHparam.mn, "SHparam.mn"); 
% printVar(SHparam.mx, "SHparam.mx");   
% printVar(SHparam.modes, "SHparam.modes");   
