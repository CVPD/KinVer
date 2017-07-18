% function accuracy = pairSVMClassification(fea, idxa, idxb, fold, matches, K, beta)
%
% Perform prediction and calculate the merged accuracy by weighing the reliability of each SVM (per feature) classifier using the beta coeficient.
%
% Input: fea; cell array that contains all the features extracted for all the pairs individuals
% Input: idxa; pairs' parent indexes. The same row of this vector and idxb's form a pair
% Input: idxb; pairs' child indexes. The same row of this vector and idxa's form a pair
% Input: fold; vector that indicates how train and test data are prepared to split in folds 
% Input: matches; class of the instances.
% Input: K; number of features
% Input: beta; Coeficient that shows the relevance of each feature. Can be one number (the same for all features) or several (in this case there is one per feature and the sum of all must equal 1).
% Output: accuracy; The accuracy calculated by the prediction on the test data for the blended classifier.

function accuracy = pairSVMClassification(fea, idxa, idxb, fold, matches, K, beta)

un = unique(fold);
nfold = length(un);

%% Perform prediction and calculate the merged accuracy by weighing the
% reliability of each SVM (per feature) classifier using the beta
% coeficient.
allScore = [];
allReal = [];
for c = 1:nfold
    trainMask = fold ~= c;
    testMask = fold == c;
    tr_idxa = idxa(trainMask);
    tr_idxb = idxb(trainMask);
    tr_matches = matches(trainMask);
    ts_idxa = idxa(testMask);
    ts_idxb = idxb(testMask);
    ts_matches = matches(testMask);
    
    currentFoldScore = zeros(length(ts_matches),1);
    
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
        currentFeatScore = predictSVMScore(svmModel,ts_Xc);
        if isreal(beta)
           betaVal = beta;
        else
            betaVal = beta{c}(p);
        end
        currentFoldScore = currentFoldScore + betaVal*currentFeatScore;
    end
    allScore = [allScore; currentFoldScore];
    allReal = [allReal; ts_matches];
end

% If predicted class are probabilities, turn into class
allScore(allScore>0) = 1;
allScore(allScore<0) = 0;
accuracy = calculateAccuracy(allReal,allScore);

end