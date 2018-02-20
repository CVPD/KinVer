% The main function that obtains results by executing the rest files
%% Configuration setting
databaseID = 'KinFaceW-II';

% Calculate features
performCalculateFeatures = false;

% Used features
useVGGFace = true;
useVGGF = true;
useLBP = false;
useHOG = false;

% Used pipeline blocks
useFeatureSelection = true;
usePCA = true;
useMNRML = true;
useLDE = false;

% Pipeline blocks configuration
% Fisher dimension reduction for KinFaceW-II
fisherDim=[ .4 .1; .2 .075; .1 .075; .075 0.1 ];
if strcmp( databaseID, 'KinFaceW-I' ) == 1
    % Fisher dimension reduction for KinFaceW-I
    fisherDim = [ .4 .075; .05 .4; .4 .1; .025 .075 ];
end
% PCA dimensions for KinFaceW-II
wdims=[ 69 66 50 62 ];
if strcmp( databaseID, 'KinFaceW-I' ) == 1
    % PCA dimensions for KinFaceW-I
    wdims = [ 66 57 48 55 ];
end
perc = 0;
T = 4;
knn = 6;
idx = 1;
K1 = 5;
K2 = 8;
sizeSVM = -1;

%% Computer paths
% Define KinFaceW database path
dbDir='../datasets';
% Define matconvnet path
convnetDir = '../matconvnet';

%% Initialization
%%% Initialization variables %%%
dbDir = strcat(dbDir,'/',databaseID);
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

%% Feature extraction
if performCalculateFeatures
    for pairIdx = 1:size(featuresFileNames,1)
        featureData = calculateSaveFeatures(imagePairsDirs(pairIdx,:), convnetDir, ...
            featuresFileNames(pairIdx,:));
        % cosineROCPlot(featuresFileName,metadataPair,pairIdStr);
        arrangeDataInPairs( featureData, metadataPairs(pairIdx,:),...
            vggFaceFileNames(pairIdx,:), vggFFileNames(pairIdx,:), ...
            LBPFileNames(pairIdx,:), HOGFileNames(pairIdx,:));
    end
end

%% Classification
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

%%% End of classification %%%
