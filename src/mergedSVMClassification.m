% function accuracy = mergedSVMClassification(mergedFeaTr, mergedFeaTs,
%   fold, matches, K, beta, sizeSVM)
%
% Performs one SVM classifier per feature and then uses the beta value for
% blending them, creating a blended classifier.
%
% Input: mergedFeaTr; individuals training data (merged pairs features) ready to perform classification per fold
% Input: mergedFeaTs; individuals testing data (merged pairs features) ready to perform classification per fold
% Input: fold; vector that indicates how train and test data are prepared to split in folds 
% Input: matches; class of the instances
% Input: K; number of features
% Input: beta; Coeficient that shows the relevance of each feature. Can be one number (the same for all features) or several (in this case there is one per feature and the sum of all must equal 1).
% Input: sizeSVM; Indicates the number of columns that the SVM uses to perform classification. It is equal to -1, all the columns are taken.
% Output: accuracy; The accuracy calculated by the prediction on the test data for the blended classifier.

function accuracy = mergedSVMClassification(mergedFeaTr, mergedFeaTs, fold, matches, K, beta, sizeSVM)

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
    tr_matches = matches(trainMask);
    ts_matches = matches(testMask);
    
    currentFoldScore = zeros(length(ts_matches),1);
    
    % Perform SVM classification
    for p = 1:K
        if sizeSVM == -1
            svmModel = trainGaussianSVM(mergedFeaTr{c}{p},tr_matches);
            currentFeatScore = predictSVMScore(svmModel,mergedFeaTs{c}{p});
        else
            svmModel = trainGaussianSVM(mergedFeaTr{c}{p}(:,1:sizeSVM),tr_matches);
            currentFeatScore = predictSVMScore(svmModel,mergedFeaTs{c}{p}(:,1:sizeSVM));
        end
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