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
classificationFileName = strcat(createdDataDir,'/','classification_',pairIdStrs,'.mat');
classificationMNRMLFileName = strcat(createdDataDir,'/','classification_MNRML_',pairIdStrs,'.mat');
vggMatFileNames = strcat(createdDataDir,'/','vgg_',pairIdStrs,'.mat');
imagenetMatFileNames = strcat(createdDataDir,'/','imagenet_',pairIdStrs,'.mat');

for pairIdx = 1:size(pairIdStrs,1)
    featFileNamesCell{pairIdx}{1} = vggMatFileNames(pairIdx,:);
    featFileNamesCell{pairIdx}{2} = imagenetMatFileNames(pairIdx,:);
end

%%% End of variables initialization %%%

%% Classification
parfor pairIdx = 1:size(featuresFileNames,1)
    [accuracy{pairIdx},accuracyMNRML{pairIdx},accuracyNRML{pairIdx}] = ...
        performClassification(imagePairsDirs(pairIdx,:), convnetDir, ...
            featuresFileNames(pairIdx,:), metadataPairs(pairIdx,:), ...
            pairIdStrs(pairIdx,:), vggMatFileNames(pairIdx,:), ...
            imagenetMatFileNames(pairIdx,:), ...
            10, 10, 40);
end
%%% End of feature extraction %%%

function [accuracy, accuracyMNRML, accuracyNRML] = performClassification(...
    imagePairsDir, convnetDir, featuresFileName, metadataPair, ...
    pairIdStr, vggMatFileName, imagenetMatFileName, T, knn, Wdims)
K = 2;
%   ;calculateSaveFeatures(imagePairsDir,convnetDir,featuresFileName);
%    cosineROCPlot(featuresFileName,metadataPair,pairIdStr);
arrangeDataInPairs(featuresFileName,metadataPair,...
    vggMatFileName,imagenetMatFileName);

load(vggMatFileName);
fea{1} = ux;
clear ux idxa idxb fold matches;
load(imagenetMatFileName);
fea{2} = ux;

% Classification on original features
accuracy = pairSVMClassification(fea, idxa, idxb, fold, matches, K, 1/K);

% Classification on MNRML
[projFea, ~, projBeta] = mnrmlProjection(fea, idxa, idxb, fold, matches, K, T, knn, Wdims);
accuracyMNRML = pairSVMClassification(projFea, idxa, idxb, fold, matches, K, projBeta);

% Classification on NRML
projFeaNRML = nrmlProjection(fea, idxa, idxb, fold, matches, K, T, knn, Wdims);
accuracyNRML = pairSVMClassification(projFeaNRML, idxa, idxb, fold, matches, K, 1/K);

end