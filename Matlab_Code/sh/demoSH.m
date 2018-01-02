% OBS(*): Data points are row vectors.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% -- (1) Create toy data -- %
% NOTE(*): Initially, Ntraining, Ntest = 3000;
% Ntraining = 3000;
% Ntest = 3000;
% NOTE(*): # groundtruth-neighbors on training set (average).
averageNumberNeighbors = 3;
% NOTE(*): Aspect ration between the two axes.
% OBS(*): Not sure why it is important.
aspectratio = 0.5;
% NOTE(*): Initially, loopbits = [2 4 8 16 32];.
loopbits = 2;

% -- (2) Add seed -- %
rng(12);

% -- (3) Import real datasets -- %
% NOTE(*): Import and normalize training set.
Xtraining = importdata('Data/Handmade/h1.train',' ',0);
Xtraining = normalize_data(Xtraining);
% NOTE(*): Import and normalize testing set.
Xtest = importdata('Data/Handmade/h1.test',' ',0);
Xtest = normalize_data(Xtest);
% NOTE(*): Define training size.
Ntraining = size(Xtraining);
Ntraining = Ntraining(1);
% NOTE(*): Define testing size.
Ntest = size(Xtest);
Ntest = Ntest(1);
% printVar(Ntraining, "Ntraining");
% printVar(Xtraining, "Xtraining");
% printVar(Xtraining(:,2), "Xtraining(:,2)");
% printVar(Xtraining, "Xtraining");
% printVar(Xtest, "Xtest");


% -- (4) Uniform distribution -- %
% NOTE(*): Both training and testing set have dimensionality d=2,
% where the second dimension is squeezed by the aspectRatio.
% OBS(*): My guess is they create a 2D dataset and try to assign it
% some structure by squeezing the 2nd dimension.
% QUESTION(*): Do you think my observation is valid? Do you think the
% aspectRatio usage here has smth to do with the aspectRatio mentioned
% in the paper?
% Xtraining = rand([Ntraining,2]);
% Xtraining(:,2) = aspectratio*Xtraining(:,2);
% Xtest = rand([Ntest,2]) ;
% Xtest(:,2) = aspectratio*Xtest(:,2);



% -- (5) Define groundtruth neighbors (used for evaluation) -- %
% OBS(*): Entry DtrueTraining(i, j) contains Eucl. distance between 
% items i and j in the training set. Also, it gets affected by 
% the aspect ratio used above (if turned on and useful, ofc).
% printVar(Xtraining, "Xtraining");
DtrueTraining = distMat(Xtraining);
% printVar(DtrueTraining, "DtrueTraining");
% printVar(Xtest, "Xtest");
% OBS(*): Entry DtrueTestTraining(i, j), contains Eucl. distance 
% between item i in testing set and item j in training set.
% NOTE(*): Size of DtrueTestTraining = [Ntest x Ntraining].
DtrueTestTraining = distMat(Xtest,Xtraining);
% printVar(DtrueTestTraining, "DtrueTestTraining");
% NOTE(*): Dball sorts elements of matrix DtrueTraining for every row.
Dball = sort(DtrueTraining,2);
% printVar(Dball, "Dball");
% printVar(Dball(:,averageNumberNeighbors), "Dball(:,averageNumberNeighbors)")
% OBS(*): % Dball is now like the average farthest distance (avg of all 
% the farthest distances for all rows, from each row to all the others 
% in the training set); So I believe this is the average Eucl. distance 
% among all the points in the training set.

% printVar(Dball(:,averageNumberNeighbors), "Dball(:,averageNumberNeighbors)");
Dball = mean(Dball(:,averageNumberNeighbors)); 
printVar(Dball, "Dball after mean");
% printVar(DtrueTestTraining, "DtrueTestTraining");
% OBS(*): % Entry WtrueTestTraining(i, j) contains 1 if item i in 
% the testing set has a smaller distance than the avg Eucl. dist 
% (calculated from the training set) to item j in the training set; 
% So I imagine somewhat like Dball is a ball around a testing point/query
% with radius defined by the training set and its avg Euclidean distance 
% among all items.
WtrueTestTraining = DtrueTestTraining < Dball; 
% printVar(WtrueTestTraining, "WtrueTestTraining");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -- (6) Demo Spectral Hashing -- %
clear SHparam score
colors = 'cbmrg'; 
i = 0;
% NOTE(*): Random test sample for visualization.
m = ceil(rand*Ntest);
% printVar(rand, "m");

for nb = loopbits
%     disp("Inside demoSH.m loop, nb = loopbits, step " + nb);
    i = i+1;
%     printVar(nb, "nb");
%     printVar(i, "i");

    % NOTE(*): #bits to code each sample.
    SHparam.nbits = nb;
    % NOTE(*): Training =>
    SHparam = trainSH(Xtraining, SHparam);
    
%     disp(Xtraining);   
    
    % NOTE(*): Compress training and testing sets.
    % OBS (*): I think U's are non-thresholded eigenfunctions, 
    % while B's are a compact repr. of the thresholded binary codes
    % (each binary code per row) found in U>0.
    [B1,U1] = compressSH(Xtraining, SHparam);
%     printVar(B1, "B1");
%     printVar(U1, "U1");
    [B2,U2] = compressSH(Xtest, SHparam);
%     printVar(B2, "B2");
%     printVar(U2, "U2");
%     printVar(U2 > 0, "U2 > 0");
%     
%     printVar(B1, "B1");
%     printVar(U1, "U1");
%     printVar(U1 > 0, "U1 > 0");

    % NOTE(*): Query example =>
%     printVar(U1>0, "U1>0");
%     printVar(U2>0, "U2>0");
    % OBS(*): Entry Dhamm(i, j) the Hamming distance from
    % item i in the testing set to item j in the training set.
    % OBS(*): Dhamm is the same as Dhamm_test defined below it.
    Dhamm = hammingDist(B2, B1);
%     Dhamm_test = distMat(U2>0, U1>0).^2;
%     printVar(Dhamm, "Dhamm between sets B2 and B1");
%     printVar(Dhamm_test, "Dhamm_test between sets B2 and B1");
    %    size(Dhamm) = [Ntest x Ntraining]

     % NOTE(*): Evaluation =>
%     printVar(WtrueTestTraining, "WtrueTestTraining");
    
    % OBS(*): Score(:,i) is assigned for each nb #bits in the initial
    % loopbits array.
    score(:,i) = evaluation(WtrueTestTraining, Dhamm, 1, 'o-', 'color', colors(i));
%     printVar(score(:,i), "score(:,i)");
%     printVar(B2, "B2(m,:)");
%     printVar(-double(hammingDist(B2(m,:), B1)'), "-double(hammingDist(B2(m,:), B1)')");
    
%     %  NOTE(*): Visualization =>
%     figure
%     subplot(211) % 211 specifies the size and position for the subplot
% %     printVar(hammingDist(B2(m,:), B1)', "hammingDist(B2(m,:), B1)");
% %     printVar(-double(hammingDist(B2(m,:), B1)'), "-double(hammingDist(B2(m,:), B1)')");
%     % Obs (*): -double(hammingDist(B2(m,:), B1)') is the minused transposed
%     % vector containing the Hamming distance from the mth item in the
%     % testing set to all the other items in the training set
%     show2dfun(Xtraining, -double(hammingDist(B2(m,:), B1)')); 
%     colormap([1 0 0; jet(nb)])
%     title({sprintf('Hamming distance to a test sample with %d bits', nb), 'red = unassigned'})
%     subplot(212)
%      % THIS THROWS ERROR and I don't know why!!!!
%     show2dfun(Xtraining, WtrueTestTraining(m,:)');
%     title('Ground truth neighbors for the test sample')
%     colormap(jet(nb))
end

% printVar(score, "final score");
% -- (7) Show eigenfunctions (I) -- %
% figure
% show2dfun(Xtraining, U1);

% printVar(loopbits, "loopbits");
% printVar(score(2,:), "score(2,:)");

% -- (8) Show eigenfunctions (II) -- %
% figure
% plot(loopbits, score(2,:))
% xlabel('number of bits')
% ylabel('precision for hamming ball 2')


% NOTE(*): Function to normalize dataset =>
% function norm_data = normalize_data(data)
%     norm_data = (data-min(data(:))) ./ (max(data(:)-min(data(:))));
% end

function norm_data = normalize_data(data)
    norm_data = (data - min(data))./(max(data) - min(data));
end





