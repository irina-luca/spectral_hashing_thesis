function [score, recall] = evaluation(Wtrue, Dhat, fig, varargin)
%
% Input:
%    Wtrue = true neighbors [Ntest * Ndataset], can be a full matrix NxN
%    Dhat  = estimated distances
%   The next inputs are optional:
%    fig = figure handle
%    options = just like in the plot command
%
% Output:
%
%               exp. # of good pairs inside hamming ball of radius <= (n-1)
%  score(n) = --------------------------------------------------------------
%               exp. # of total pairs inside hamming ball of radius <= (n-1)
%
%               exp. # of good pairs inside hamming ball of radius <= (n-1)
%  recall(n) = --------------------------------------------------------------
%                          exp. # of total good pairs 

disp("# -- evaluation.m called -- #");

% printVar(Wtrue, "Wtrue");

[Ntest, Ntrain] = size(Wtrue);

% OBS(*): Wtrue(:) just stacks Wtrue's columns from top to bottom in a 
% long line.
% OBS(*): total_good_pairs of items j in the training set
% that are within the avg Euclidean distance with respect to each(all)
% item/query in the testing set.
total_good_pairs = sum(Wtrue(:));

% printVar(total_good_pairs, "total_good_pairs");

% NOTE(*): Find pairs with similar codes.
score = zeros(20,1);
% OBS(*): I defined this myself!!!!
recall = zeros(20,1);
% printVar(score, "score");
% printVar(Wtrue, "Wtrue");
% OBS(*): n = 1 -> 20
for n = 1:length(score)
%     disp("# -- evaluation.m => iteration " + n + " -- #");
    % OBS(*): j will return the indices of the items in Dhat that 
    % satisfy the predicate, but in a column-wise fashion
%     printVar(((n-1)+0.00001), "((n-1)+0.00001)");
%     printVar(Dhat, "Dhat");
    j = find(Dhat<=((n-1)+0.00001));
%     printVar(j, "j");
%     printVar(Wtrue(j), "Wtrue(j)");

    %exp. # of good pairs that have exactly the same code
    retrieved_good_pairs = sum(Wtrue(j));
    
    % exp. # of total pairs that have exactly the same code
    % OBS(*): I guess this is like # retrieved pairs in Hamming
    retrieved_pairs = length(j);

%     printVar(retrieved_good_pairs, "retrieved_good_pairs");
%     printVar(retrieved_pairs, "retrieved_pairs");
%     printVar(total_good_pairs, "total_good_pairs");
    score(n) = retrieved_good_pairs/retrieved_pairs;
    recall(n)= retrieved_good_pairs/total_good_pairs;
end
 
printVar(score, "score");
printVar(recall, "recall");

% The standard measures for IR are recall and precision. Assuming that:
%
%    * RET is the set of all items the system has retrieved for a specific inquiry;
%    * REL is the set of relevant items for a specific inquiry;
%    * RETREL is the set of the retrieved relevant items 
%
% then precision and recall measures are obtained as follows:
%
%    precision = RETREL / RET
%    recall = RETREL / REL 


% 
% if nargout == 0 || nargin > 3
%     if isempty(fig);
%         fig = figure;
%     end
%     figure(fig)
%     subplot(211)
%     plot(0:length(score)-1, score, varargin{:})
%     hold on
%     xlabel('hamming radium')
%     ylabel('percent correct (precision)')
%     title('percentage of good neighbors inside the hamm ball')
%     
% %     subplot(211)
% %     plot(0:length(recall)-1, recall, varargin{:})
% %     hold on
% %     xlabel('hamming radium')
% %     ylabel('percent correct (recall)')
% %     title('percentage of good neighbors inside the hamm ball')
%     
%     subplot(212)
%     plot(recall, score, varargin{:})
%     hold on
%     axis([0 1 0 1])
%     xlabel('recall')
%     ylabel('percent correct (precision)')
%     drawnow
% end
