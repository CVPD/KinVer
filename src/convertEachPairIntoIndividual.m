% function [mergedFeaTr, mergedFeaTs] = convertEachPairIntoIndividual(fea, idxa, idxb, fold, K)
%
% Merges the data that is arranged in parents and children in pairs.
%
% Input: fea; cell array that contains all the features extracted for all the pairs individuals
% Input: idxa; pairs' parent indexes. The same row of this vector and idxb's form a pair
% Input: idxb; pairs' child indexes. The same row of this vector and idxa's form a pair
% Input: fold; vector that indicates how train and test data are prepared to split in folds 
% Input: K; number of features
% Output: mergedFeaTr; individuals training data (merged pairs features) ready to perform classification per fold
% Output: mergedFeaTs; individuals testing data (merged pairs features) ready to perform classification per fold
function [mergedFeaTr, mergedFeaTs] = convertEachPairIntoIndividual(fea, idxa, idxb, fold, K)

un = unique(fold);
nfold = length(un);

for c = 1:nfold
    trainMask = fold ~= c;
    testMask = fold == c;
    tr_idxa = idxa(trainMask);
    tr_idxb = idxb(trainMask);
    ts_idxa = idxa(testMask);
    ts_idxb = idxb(testMask);
    
    % Merge parent and child 2 feature vectors into one (Train data)
    for p = 1:K
        if iscell(fea{1}) % Features per fold
            X = fea{c}{p};
        else
            X = fea{p}; % Features (no folds)
        end
        tr_Xa = X(tr_idxa, :);                    % training data
        tr_Xb = X(tr_idxb, :);                    % training data
        mergedFeaTr{c}{p} = mergePairsInMatrix(tr_Xa, tr_Xb);
        
        ts_Xa = X(ts_idxa, :);                 % testing data
        ts_Xb = X(ts_idxb, :);                 % testing data
        mergedFeaTs{c}{p} = mergePairsInMatrix(ts_Xa, ts_Xb);
    end
end

end