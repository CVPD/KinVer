%% Initialization
clear all;
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
    getClassificationData(featFileNamesCell{idx}, classificationFileName(idx,:))
    mnrmlSpaceChange(classificationFileName(idx,:), ...
       classificationMNRMLFileName(idx,:));
   accuracy{idx} = pairSVMClassification(classificationFileName(idx,:)); 
   accuracyMNRML{idx} = pairSVMClassification(classificationMNRMLFileName(idx,:));
end
%%% End of feature extraction %%%
