%% Configuration setting
databaseID = 'KinFaceW-II';

performCalculateFeatures = false;

% Features setting
useVGGFace = true;
useVGGF = true;
useLBP = false;
useHOG = false;

% Pipeline setting
useFeatureSelection = true;
usePCA = true;
useMNRML = true;
useLDE = false;

%% Computer paths
% Define KinFaceW database path
dbDir='C:\Users\windows\Desktop\TFM\datasets';
% Define matconvnet path
convnetDir = 'C:\Users\windows\Desktop\TFM\matconvnet-1.0-beta23';

%% Initialization
%%% Initialization variables %%%
strcat(dbDir,'/',databaseID)
% Database specific (configured for KinFaceW-I and KinFaceW-II
metadataDir = strcat(dbDir,'/','meta_data');
imagePairsDirs = ['father-dau';'father-son';'mother-dau';'mother-son'];
imagePairsDirs = strcat(dbDir,'/images','/',imagePairsDirs);
pairIdStrs = ['fd';'fs';'md';'ms'];
metadataPairs = strcat(metadataDir,'/',pairIdStrs,'_pairs.mat');

% Names of the files created by these scripts
parentDir = cd(cd('..'));
createdDataDir = strcat(parentDir,'/','data','/','data-',databaseID);
featuresFileNames = strcat(createdDataDir,'/',pairIdStrs,'-features.mat');
vggFaceFileNames = strcat(createdDataDir,'/','vggFace_',pairIdStrs,'.mat');
vggFFileNames = strcat(createdDataDir,'/','vggF_',pairIdStrs,'.mat');
LBPFileNames = strcat(createdDataDir,'/','LBP_',pairIdStrs,'.mat');
HOGFileNames = strcat(createdDataDir,'/','HOG_',pairIdStrs,'.mat');

for pairIdx = 1:size(pairIdStrs,1)
    featFileNamesCell{pairIdx}{1} = vggFaceFileNames(pairIdx,:);
    featFileNamesCell{pairIdx}{2} = vggFFileNames(pairIdx,:);
    featFileNamesCell{pairIdx}{3} = LBPFileNames(pairIdx,:);
    featFileNamesCell{pairIdx}{4} = HOGFileNames(pairIdx,:);
end

%%% End of variables initialization %%%

%% Classification
T = 4;
knn = 6;
idx = 1;
range = 15:70;%20:40;
perc = 0;
wdims=[60 69 61 58];
fisherDim=[0.075 0.4; 0.05 0.05; 0.05 0.05; 0.075 0.075];
%for wdims = range
%for K1 = 4:6
%    for K2 = 2:10
K1 = 5;
K2 = 8;
%range = 5:5:wdims;%20:40;

% Add wdims if it is not in the range
%if isempty(find(range==wdims))
%    range(length(range)+1) = wdims;
%end
sizeSVM = -1;
%for sizeSVM = range
    for pairIdx = 1:size(featuresFileNames,1)
        
        [accuracy(pairIdx),accuracyMNRML(pairIdx),accuracyNRML(pairIdx), ...
            accuracyPerFeat(pairIdx,:), numEigVals(pairIdx,idx),betaPerFeat(pairIdx,:)] = ...
            performClassification(imagePairsDirs(pairIdx,:), convnetDir, ...
            featuresFileNames(pairIdx,:), metadataPairs(pairIdx,:), ...
            pairIdStrs(pairIdx,:), vggFaceFileNames(pairIdx,:), ...
            vggFFileNames(pairIdx,:), LBPFileNames(pairIdx,:), ...
            HOGFileNames(pairIdx,:), ...
            T, knn, perc, K1, K2, wdims(pairIdx),sizeSVM,fisherDim(pairIdx,:),...
            performCalculateFeatures, ...
            useVGGFace, useVGGF, useLBP, useHOG, ...
            useFeatureSelection, usePCA, useMNRML, useLDE);
        
    end
    meanAccuracy(idx) = mean(accuracyMNRML);
    accuracyMNRMLIdx(idx,:) = accuracyMNRML;
    betaMeans(idx,:) = mean(betaPerFeat,1);
    idx = idx+1;
%end
%    end
%end
%end
%plot(mean(numEigVals),meanAccuracy);
%title('Accuracy/Number eigenvalues');
%xlabel('Number eigenvalues');
%ylabel('Accuracy');
%plot(range,meanAccuracy);
%title('Accuracy/LDE vector dim');
%xlabel('LDE vector dim');
%ylabel('Accuracy');
%%% End of classification %%%

function [accuracy, accuracyMNRML, accuracyNRML, accuracyPerFeat, ...
    numEigvals, betaMeans] = performClassification(...
    imagePairsDir, convnetDir, featuresFileName, metadataPair, ...
    pairIdStr, vggMatFileName, imagenetMatFileName, ...
    LBPMatFileName, HOGMatFileName, T, knn, eigValPerc, ...
    K1, K2, wdims, sizeSVM,feaSelectionDims, performCalculateFeatures, ...
    useVGGFace, useVGGF, useLBP, useHOG, ...
    useFeatureSelection, usePCAprojection, useMNRMLprojection, useLDEprojection)
accuracy = 0; accuracyMNRML = 0; accuracyNRML = 0; accuracyPerFeat = 0;
numEigvals = 0;

if performCalculateFeatures
    calculateSaveFeatures(imagePairsDir,convnetDir,featuresFileName);
    arrangeDataInPairs(featuresFileName,metadataPair,...
        vggMatFileName,imagenetMatFileName, LBPMatFileName, HOGMatFileName);
end
% cosineROCPlot(featuresFileName,metadataPair,pairIdStr);

numFeats = 0;
if useVGGFace
    clear ux idxa idxb fold matches;
    load(vggMatFileName);
    fea{numFeats+1} = ux;
    numFeats = numFeats + 1;
end
if useVGGF
    clear ux idxa idxb fold matches;
    load(imagenetMatFileName);
    fea{numFeats+1} = ux;
    numFeats = numFeats + 1;
end
if useLBP
    clear ux idxa idxb fold matches;
    load(LBPMatFileName);
    fea{numFeats+1} = ux;
    numFeats = numFeats + 1;
end
if useHOG
    clear ux idxa idxb fold matches;
    load(HOGMatFileName);
    fea{numFeats+1} = ux;
    numFeats = numFeats + 1;
end
if numFeats == 0
    error('number of features must be bigger than 0');
end
K = numFeats;

% Classification on original features
%accuracy = pairSVMClassification(fea, idxa, idxb, fold, matches, K, 1/K);
accuracy = 0; accuracyPerFeat = 0;
%accuracyPerFeat = pairSVMClassificationPerFeat(fea, idxa, idxb, fold, matches, K);

un = unique(fold);
nfold = length(un);

% Classification on MNRML
if useFeatureSelection
    fea = feaSelectionFisherMerge(fea, idxa, idxb, fold, ... % feaSelectionFisherMerge
        matches, K, feaSelectionDims);%feaSelectionVariance(fea, K);
end

if usePCAprojection && useMNRMLprojection
    [fea, ~, projBeta] = PCAplusMNRMLprojections(fea, idxa, idxb, fold, ...
        matches, K, T, knn, eigValPerc, wdims);
    betasVec = cell2mat(projBeta);
    betasMat = transpose(reshape(betasVec,[K nfold]));
    betaMeans = mean(betasMat,1);
else
    if usePCAprojection
        fea = mnrmlProjection(fea, idxa, idxb, fold, ...
            matches, K, eigValPerc, wdims);
    else
        [fea, ~, projBeta] = mnrmlProjection(fea, idxa, idxb, fold, ...
        matches, K, T, knn, eigValPerc, wdims);
        betasVec = cell2mat(projBeta);
        betasMat = transpose(reshape(betasVec,[K nfold]));
        betaMeans = mean(betasMat,1);
    end    
end

numEigvals = size(fea{1}{1},2); % Wdims
[mergedFeaTr, mergedFeaTs]= convertEachPairIntoIndividual(fea, idxa, idxb, fold, K);

if useLDEprojection
    [mergedFeaTr, mergedFeaTs]=ldeProjection(mergedFeaTr, mergedFeaTs, fold, matches, K, K1, K2);
end

accuracyMNRML = mergedSVMClassification(mergedFeaTr, mergedFeaTs, fold, matches, K, projBeta, sizeSVM);

% Classification on NRML
%Wdims = 30;
%projFeaNRML = nrmlProjection(fea, idxa, idxb, fold, matches, K, T, knn, Wdims);
%[mergedFeaTrNRML, mergedFeaTsNRML]= convertEachPairIntoIndividual(projFeaNRML, idxa, idxb, fold, K);
%accuracyNRML = mergedSVMClassification(mergedFeaTrNRML, mergedFeaTsNRML, fold, matches, K, 1/K);

end