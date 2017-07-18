% function fea = feaSelectionFisherMerge(fea, idxa, idxb, fold, ...
%    matches, K ,numSelect)
%
% Selects the most relevant descriptors of the input features using fisher
% feature selection (supervised)
%
% Input: fea; cell array that contains all the features extracted for all the pairs individuals
% Input: idxa; pairs' parent indexes. The same row of this vector and idxb's form a pair
% Input: idxb; pairs' child indexes. The same row of this vector and idxa's form a pair
% Input: fold; vector that indicates how train and test data are prepared to split in folds
% Input: matches; class of the instances
% Input: K; number of features
% Input: numSelect; the number of descrptors that are selected for each feature
% Output: fea; input features after selecting the most relevant descriptors
function fea = feaSelectionFisherMerge(fea, idxa, idxb, fold, ...
    matches, K ,numSelect)

addpath('./external/fisher/lib'); % dependencies
addpath('./external/fisher/methods'); % FS methods

un = unique(fold);
nfold = length(un);

c = 1;% chosen train data is of fold 1
trainMask = fold ~= c;
testMask = fold == c;
tr_idxa = idxa(trainMask);
tr_idxb = idxb(trainMask);
tr_matches = matches(trainMask);
ts_idxa = idxa(testMask);
ts_idxb = idxb(testMask);
ts_matches = matches(testMask);

for p = 1:K
    X = fea{p};
    tr_Xa = X(tr_idxa, :);                    % training data
    tr_Xb = X(tr_idxb, :);                    % training data
    tr_Xc = mergePairsInMatrix(tr_Xa, tr_Xb);
    
    numF = size(tr_Xc,2);
    tr_matches = tr_matches + 0; % turn into double
    tr_matches(tr_matches==0,:) = -1;
    ranking = spider_wrapper(tr_Xc,tr_matches,numF,'fisher');
    
    numSelectedFeats = round(numF*numSelect(p));
    if numSelectedFeats > 49
        X = X(:,ranking(1:numSelectedFeats));
        fea{p} = X;
        clear X;
    end
end
end
