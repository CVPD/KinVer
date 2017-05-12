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
vggMatFileName = strcat(createdDataDir,'/','vgg_',pairIdStrs,'.mat');
imagenetMatFileName = strcat(createdDataDir,'/','imagenet_',pairIdStrs,'.mat');

for idx = 1:size(pairIdStrs,1)
    featFileNamesCell{idx}{1} = vggMatFileName(idx,:);
    featFileNamesCell{idx}{2} = imagenetMatFileName(idx,:);
end

%%% End of variables initialization %%%

%% Feature extraction
for idx = 1:size(featuresFileNames,1)
 %   calculateSaveFeatures(imagePairsDirs(idx,:),convnetDir,featuresFileNames(idx,:));
%    cosineROCPlot(featuresFileNames(idx,:),metadataPairs(idx,:),pairIdStrs(idx,:));
    arrangeDataInPairs(featuresFileNames(idx,:),metadataPairs(idx,:),...
        vggMatFileName(idx,:),imagenetMatFileName(idx,:));
    
    load(vggMatFileName(idx,:));
    fea{1} = ux;
    clear ux idxa idxb fold matches;
    load(imagenetMatFileName(idx,:));
    fea{2} = ux;
    
    T = 1;        % Iterations
    knn = 5;      % k-nearest neighbors
    Wdims = 40;  % low dimension
    K = 2;
    
    % Classification on original features
    %accuracy{idx} = pairSVMClassification(fea, idxa, idxb, fold, matches, K, 1/K);
    
    % Classification on MNRML
    [projFea, ~, projBeta] = mnrmlProjection(fea, idxa, idxb, fold, matches, K, T, knn, Wdims);
    accuracyMNRML{idx} = pairSVMClassification(projFea, idxa, idxb, fold, matches, K, projBeta);
    
    % Classification on NRML
    %projFeaNRML = nrmlProjection(fea, idxa, idxb, fold, matches, K, T, knn, Wdims);
    %accuracyNRML{idx} = pairSVMClassification(projFeaNRML, idxa, idxb, fold, matches, K, 1/K);
    
end
%%% End of feature extraction %%%
