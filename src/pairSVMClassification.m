% inputFile = 'C:\Users\oscar\Desktop\TFM\project\data\mnrmlFeat_ms.mat';
% inputFile = 'C:\Users\oscar\Desktop\TFM\project\data\mnrmlFeat_ms_noW.mat';
% SVMClassification(inputFile);

function accuracy = pairSVMClassification(inputFile)

load(inputFile);

% Merge parent and child 2 feature vectors into one (Train data)
Xtrc = mergeIndividualsPerFeaturesAndFolds(Xtra,Xtrb);

% Merge parent and child 2 feature vectors into one (Test data)
Xtsc = mergeIndividualsPerFeaturesAndFolds(Xtsa,Xtsb);

numFolds = size(Xtrc,2);
numFeat = size(Xtrc{1},1);

%% Perform prediction and calculate the merged accuracy by weighing the 
% reliability of each SVM (per feature) classifier using the beta 
% coeficient.
allScore = [];
allReal = [];
for fold = 1:numFolds
    currentFoldScore = zeros(length(ts_matches{fold}),1);
    for feat = 1:numFeat
        svmModel = trainLinearSVM(Xtrc{fold}{feat},tr_matches{fold});
        currentFeatScore = predictSVMScore(svmModel,Xtsc{fold}{feat});
        currentFoldScore = currentFoldScore + beta{fold}(feat)*currentFeatScore;
    end
    allScore = [allScore; currentFoldScore];
    allReal = [allReal; ts_matches{fold}];
end


% If predicted class are probabilities, turn into class
allScore(allScore>0) = 1;
allScore(allScore<0) = 0;
accuracy = calculateAccuracy(allReal,allScore);
end