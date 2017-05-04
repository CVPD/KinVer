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

for feat = 1:numFeat
    allReal{feat} = [];
    allScore{feat} = [];
end

for fold = 1:numFolds
    for feat = 1:numFeat
        svmModel = trainLinearSVM(Xtrc{fold}{feat},tr_matches{fold});
        score = predictSVMScore(svmModel,Xtsc{fold}{feat});
        allReal{feat} = [allReal{feat};ts_matches{fold}];
        allScore{feat} = [allScore{feat};score];
    end
end

%% Calculate accuracy a SVM per feature individually (comment to remove)
% for feat = 1:numFeat
%     accuracy{feat} = calculateAccuracy(allReal{feat},allPredicted{feat});
% end

%% Calculate the merged accuracy by weighing the reliability of each SVM 
%(per feature) classifier using the beta coeficient. (comment to remove)
allScoreMerged = zeros(length(allScore{1}),1);
for f = 1:numFeat
    allScoreMerged = allScoreMerged + beta{f}*allScore{f};
end
% If predicted class are probabilities, turn into class
allScoreMerged(allScoreMerged>0) = 1;
allScoreMerged(allScoreMerged<0) = 0;
accuracy = calculateAccuracy(allReal{1},allScoreMerged);
end