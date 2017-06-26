%% Initialization
%clear all;
%%% PC specific %%%
% Define KinFaceW database path
dbDir='C:\Users\oscar\Desktop\TFM\datasets\KinFaceW-I';
% Define matconvnet path
convnetDir = 'C:\Users\oscar\Desktop\TFM\matconvnet-1.0-beta23';
%%% End PC specific %%%

%%% Initialization variables %%%
% Database specific (configured for KinFaceW-I and KinFaceW-II
metadataDir = strcat(dbDir,'/','meta_data');
imagePairsDirs = ['father-dau';'father-son';'mother-dau';'mother-son'];
imagePairsDirs = strcat(dbDir,'/images','/',imagePairsDirs);
pairIdStrs = ['fd';'fs';'md';'ms'];
metadataPairs = strcat(metadataDir,'/',pairIdStrs,'_pairs.mat');

% Names of the files created by these scripts
parentDir = cd(cd('..'));
createdDataDir = strcat(parentDir,'/','data');
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
%range = 17:38;%20:40;
perc = 0;
wdims=[45 64 51 40];
fisherDim=[0.0750 0.4000;0.0500 0.0500;0.0500 0.0500;0.0750 0.0750];
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
            T, knn, perc, K1, K2, wdims(pairIdx),sizeSVM,fisherDim(pairIdx,:));
        
    end
    meanAccuracy(idx) = mean(accuracyMNRML);
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
    K1, K2, wdims, sizeSVM,feaSelectionDims)
accuracy = 0; accuracyMNRML = 0; accuracyNRML = 0; accuracyPerFeat = 0;
numEigvals = 0;
%calculateSaveFeatures(imagePairsDir,convnetDir,featuresFileName);
% cosineROCPlot(featuresFileName,metadataPair,pairIdStr);
%arrangeDataInPairs(featuresFileName,metadataPair,...
%    vggMatFileName,imagenetMatFileName, LBPMatFileName, HOGMatFileName);

load(vggMatFileName);
fea{1} = ux;
clear ux idxa idxb fold matches;
load(imagenetMatFileName);
fea{2} = ux;
%clear ux idxa idxb fold matches;
%load(LBPMatFileName);
%fea{3} = ux;
%clear ux idxa idxb fold matches;
%load(HOGMatFileName);
%fea{4} = ux;
K = 2;
% Clas  sification on original features
accuracy = pairSVMClassification(fea, idxa, idxb, fold, matches, K, 1/K);

accuracyPerFeat = pairSVMClassificationPerFeat(fea, idxa, idxb, fold, matches, K);

un = unique(fold);
nfold = length(un);

% Classification on MNRML
fea = feaSelectionFisherMerge(fea, idxa, idxb, fold, ...
    matches, K, feaSelectionDims);%feaSelectionVariance(fea, K);
[projFea, ~, projBeta] = mnrmlProjection(fea, idxa, idxb, fold, ...
    matches, K, T, knn, eigValPerc, wdims);
betasVec = cell2mat(projBeta);
betasMat = transpose(reshape(betasVec,[K nfold]));
betaMeans = mean(betasMat,1);

numEigvals = size(projFea{1}{1},2); % Wdims
[mergedFeaTr, mergedFeaTs]= convertEachPairIntoIndividual(projFea, idxa, idxb, fold, K);
%[mergedFeaTr, mergedFeaTs]=ldeProjection(mergedFeaTr, mergedFeaTs, fold, matches, K, K1, K2);
accuracyMNRML = mergedSVMClassification(mergedFeaTr, mergedFeaTs, fold, matches, K, projBeta, sizeSVM);

% Classification on NRML
%Wdims = 30;
%projFeaNRML = nrmlProjection(fea, idxa, idxb, fold, matches, K, T, knn, Wdims);
%[mergedFeaTrNRML, mergedFeaTsNRML]= convertEachPairIntoIndividual(projFeaNRML, idxa, idxb, fold, K);
%accuracyNRML = mergedSVMClassification(mergedFeaTrNRML, mergedFeaTsNRML, fold, matches, K, 1/K);

end