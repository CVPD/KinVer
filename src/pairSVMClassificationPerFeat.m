% function accuracy = pairSVMClassificationPerFeat(fea, idxa, idxb, fold, matches, K)
%
% Perform prediction and calculate the accuracy of each SVM (one per feature) classifier.
%
% Input: fea; cell array that contains all the features extracted for all the pairs individuals
% Input: idxa; pairs' parent indexes. The same row of this vector and idxb's form a pair
% Input: idxb; pairs' child indexes. The same row of this vector and idxa's form a pair
% Input: fold; vector that indicates how train and test data are prepared to split in folds 
% Input: matches; class of the instances.
% Input: K; number of features
% Output: accuracy; The accuracy calculated by the prediction on the test
% data for each classifier (one per feature).

function accuracy = pairSVMClassificationPerFeat(fea, idxa, idxb, fold, matches, K)

un = unique(fold);
nfold = length(un);

allReal = [];
for p = 1:K
    allScore{p} = [];
end

%% Perform prediction and calculate the merged accuracy by weighing the
% reliability of each SVM (per feature) classifier using the beta
% coeficient.
for c = 1:nfold
    trainMask = fold ~= c;
    testMask = fold == c;
    tr_idxa = idxa(trainMask);
    tr_idxb = idxb(trainMask);
    tr_matches = matches(trainMask);
    ts_idxa = idxa(testMask);
    ts_idxb = idxb(testMask);
    ts_matches = matches(testMask);
    
    % Merge parent and child 2 feature vectors into one (Train data)
    for p = 1:K
        if iscell(fea{1}) % Features per fold
            X = fea{c}{p};
        else
            X = fea{p}; % Features (no folds)
        end
        tr_Xa = X(tr_idxa, :);                    % training data
        tr_Xb = X(tr_idxb, :);                    % training data
        tr_Xc = mergePairsInMatrix(tr_Xa, tr_Xb);
        
        ts_Xa = X(ts_idxa, :);                 % testing data
        ts_Xb = X(ts_idxb, :);                 % testing data
        ts_Xc = mergePairsInMatrix(ts_Xa, ts_Xb);
        
        svmModel = trainGaussianSVM(tr_Xc,tr_matches);
        currentFoldFeatScore = predictSVMScore(svmModel,ts_Xc);
        allScore{p} = [allScore{p}; currentFoldFeatScore];
    end
    allReal = [allReal; ts_matches];
end

% If predicted class are probabilities, turn into class
for p = 1:K
    allScore{p}(allScore{p}>0) = 1;
    allScore{p}(allScore{p}<0) = 0;
    accuracy{p} = calculateAccuracy(allReal,allScore{p});
end
    
end