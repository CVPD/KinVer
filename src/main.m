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

for pairIdx = 1:size(pairIdStrs,1)
    featFileNamesCell{pairIdx}{1} = vggFaceFileNames(pairIdx,:);
    featFileNamesCell{pairIdx}{2} = vggFFileNames(pairIdx,:);
end

%%% End of variables initialization %%%

%% Classification
T = 4;
knn = 6;
idx = 1;
range = 0.1:0.1:1;
%perc = 0.65;
for perc = range
%for K1 = 2:10
%    for K2 = 2:10
K1 = 5;
K2 = 5;
    for pairIdx = 1:size(featuresFileNames,1)
            
            [accuracy(pairIdx),accuracyMNRML(pairIdx),accuracyNRML(pairIdx), ...
                accuracyPerFeat(pairIdx,:), numEigVals(idx)] = ...
                performClassification(imagePairsDirs(pairIdx,:), convnetDir, ...
                featuresFileNames(pairIdx,:), metadataPairs(pairIdx,:), ...
                pairIdStrs(pairIdx,:), vggFaceFileNames(pairIdx,:), ...
                vggFFileNames(pairIdx,:), ...
                T, knn, perc, K1, K2);
            
        end
        meanAccuracy(idx) = mean(accuracyMNRML);
        idx = idx+1;        
%    end
%end
end
plot(numEigVals,meanAccuracy);
title('Accuracy/Number eigenvalues');
xlabel('Number eigenvalues');
ylabel('Accuracy');
%%% End of classification %%%

function [accuracy, accuracyMNRML, accuracyNRML, accuracyPerFeat, ...
    numEigvals] = performClassification(...
    imagePairsDir, convnetDir, featuresFileName, metadataPair, ...
    pairIdStr, vggMatFileName, imagenetMatFileName, T, knn, eigValPerc, ...
    K1, K2)
K = 2;
accuracy = 0; accuracyMNRML = 0; accuracyNRML = 0; accuracyPerFeat = 0;
numEigvals = 0;
%calculateSaveFeatures(imagePairsDir,convnetDir,featuresFileName);
% cosineROCPlot(featuresFileName,metadataPair,pairIdStr);
%arrangeDataInPairs(featuresFileName,metadataPair,...
%    vggMatFileName,imagenetMatFileName);

load(vggMatFileName);
fea{1} = ux;
clear ux idxa idxb fold matches;
load(imagenetMatFileName);
fea{2} = ux;

% Classification on original features
%accuracy = pairSVMClassification(fea, idxa, idxb, fold, matches, K, 1/K);

%accuracyPerFeat = pairSVMClassificationPerFeat(fea, idxa, idxb, fold, matches, K);

% Classification on MNRML
[projFea, ~, projBeta] = mnrmlProjection(fea, idxa, idxb, fold, matches, K, T, knn, eigValPerc);
numEigvals = size(projFea{1}{1},2); % Wdims
[mergedFeaTr, mergedFeaTs]= convertEachPairIntoIndividual(projFea, idxa, idxb, fold, K);
[mergedFeaTr, mergedFeaTs]=ldeProjection(mergedFeaTr, mergedFeaTs, fold, matches, K, K1, K2);
accuracyMNRML = mergedSVMClassification(mergedFeaTr, mergedFeaTs, fold, matches, K, projBeta);

% Classification on NRML
%Wdims = 30;
%projFeaNRML = nrmlProjection(fea, idxa, idxb, fold, matches, K, T, knn, Wdims);
%[mergedFeaTrNRML, mergedFeaTsNRML]= convertEachPairIntoIndividual(projFeaNRML, idxa, idxb, fold, K);
%accuracyNRML = mergedSVMClassification(mergedFeaTrNRML, mergedFeaTsNRML, fold, matches, K, 1/K);

end