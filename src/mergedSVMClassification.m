% inputFile = 'C:\Users\oscar\Desktop\TFM\project\data\mnrmlFeat_ms.mat';
% inputFile = 'C:\Users\oscar\Desktop\TFM\project\data\mnrmlFeat_ms_noW.mat';
% SVMClassification(inputFile);

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