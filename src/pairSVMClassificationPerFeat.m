% inputFile = 'C:\Users\oscar\Desktop\TFM\project\data\mnrmlFeat_ms.mat';
% inputFile = 'C:\Users\oscar\Desktop\TFM\project\data\mnrmlFeat_ms_noW.mat';
% SVMClassification(inputFile);

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