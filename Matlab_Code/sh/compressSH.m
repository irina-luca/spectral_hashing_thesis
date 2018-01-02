function [B, U] = compressSH(X, SHparam)
%
% [B, U] = compressSH(X, SHparam)
%
% Input
%   X = features matrix [Nsamples, Nfeatures]
%   SHparam =  parameters (output of trainSH)
%
% Output
%   B = bits (compacted in 8 bits words)
%   U = value of eigenfunctions (bits in B correspond to U>0)

disp("# -- compressSH called -- #");

[Nsamples Ndim] = size(X);
nbits = SHparam.nbits;

% printVar(X, "X");
% printVar(SHparam.pc, "SHparam.pc");

% OBS(*): We still transform the data and change the basis. Result seems
% the same as in trainSH.m.
X = X*SHparam.pc;
% printVar(X, "X after X = X*SHparam.pc");

% printVar(repmat(SHparam.mn, [Nsamples 1]), "repmat(SHparam.mn, [Nsamples 1])");

% OBS(*): This is X PCA-ed after substracting the min of each pc/dim.
% from it.
X = X-repmat(SHparam.mn, [Nsamples 1]);
% disp("[X PCA-ed after substracting the min of each pca/dim from it]");
% printVar(X, "X after X = X-repmat(SHparam.mn, [Nsamples 1])");

% OBS(*): This is the same omega0 as in trainSH.m, where
% (SHparam.mx-SHparam.mn) = R, s.t. R contains ranges for each dim./pc.
% OBS(*): omega0 seems to be pi/(b - a) from the eigenfunction formula
% in the paper.
omega0=pi./(SHparam.mx-SHparam.mn);
% printVar(omega0, "same omega0");
% printVar(repmat(omega0, [nbits 1]), "repmat(omega0, [nbits 1])");
omegas=SHparam.modes.*repmat(omega0, [nbits 1]);
% printVar(omegas, "omegas");
% printVar(omegas(1,:), "omegas(1,:)");
% printVar(omegas(2,:), "omegas(2,:)");
U = zeros([Nsamples nbits]);
% printVar(U, "U");
for i=1:nbits
%     disp("### Inside i=1:nbits loop ###, step " + i);
    % OBS(*): omegas(i,:) takes ith row of omegas and omegai
    % is the result of repeating it for #Nsamples
    omegai = repmat(omegas(i,:), [Nsamples 1]);
%     printVar(omegai, "omegai");
%     printVar(X, "X");
%     printVar(X.*omegai, "X.*omegai");
    ys = sin(X.*omegai+pi/2);
%     printVar(ys, "ys");
    
    % OBS(*): yi is a vector with each entry being the prod on that row 
    % for the elements on the same row in ys.
    yi = prod(ys,2);
%     printVar(yi, "yi");
    U(:,i)=yi;    
end

% OBS(*): U contains on each column the results after sin() for each bit
printVar(U, "U");
% printVar(U>0, "U>0");

B = compactbit(U>0);

% printVar(B, "B");


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% So far, I don t get anything from what this function means to do!!!!,
% but I only have a guess: it might convert the thresholded (U>0) to
% compact representation for each and every row
function cb = compactbit(b)
%
% b = bits array
% cb = compacted string of bits (using words of 'word' bits)

% printVar(b, "b from 'compactbit(b)', where b = (U > 0)");

[nSamples nbits] = size(b);
% disp(nSamples);
% disp(nbits);
% OBS(*): nwords says what multiple of 8 bits we have, so #words
nwords = ceil(nbits/8); 
% disp(nwords);
cb = zeros([nSamples nwords], 'uint8');
% printVar(cb, "cb");

for j = 1:nbits
%     disp("From compressSH.m, j = 1:nbits, #iteration " + j);
    
    % Obs (*): w probably tells in which words bit j is contained
    w = ceil(j/8); 
%     if j == 2
%         printVar(w, "w");
%         printVar(cb(:,w), "cb(:,w)");
%         printVar(mod(j-1,8)+1, "mod(j-1,8)+1");
%         printVar(b(:,j), "b(:,j)");
%     end
        % Obs (*): From docs, intout = bitset(A,bit,V) 
        % returns A with position bit set to the value of V.
        % => 
        cb(:,w) = bitset(cb(:,w), mod(j-1,8)+1, b(:,j));
    
%     printVar(cb(:,w), "cb(:,w) after assigned in iteration " + j);
    
end


