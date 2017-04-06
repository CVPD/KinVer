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
mergedVggFileName = strcat(createdDataDir,'/','merged_vgg_',pairIdStrs,'.csv');
mergedImagenetFileName = strcat(createdDataDir,'/','merged_imagenet_',pairIdStrs,'.csv');
vggMatFileName = strcat(createdDataDir,'/','vgg_',pairIdStrs,'.mat');
imagenetMatFileName = strcat(createdDataDir,'/','imagenet_',pairIdStrs,'.mat');
numFeatures = 2;

accuracyCell = {};
for idx = 1:size(pairIdStrs,1)
    accuracyCellArray{idx}.pair = pairIdStrs(idx,:);
end
%%% End of variables initialization %%%

%%% Feature extraction %%%
for idx = 1:size(featuresFileNames,1)
%     calculateSaveVGGFeatures(imagePairsDirs(idx,:),convnetDir,featuresFileNames(idx,:));
%     cosineROCPlot(featuresFileNames(idx,:),metadataPairs(idx,:),pairIdStrs(idx,:));
%     exportFeaturesToCSV(featuresFileNames(idx,:),metadataPairs(idx,:),...
%         mergedVggFileName(idx,:),mergedImagenetFileName(idx,:));
saveFeaturesInPairsMat(featuresFileNames(idx,:),metadataPairs(idx,:),...
         vggMatFileName(idx,:),imagenetMatFileName(idx,:));
end
%%% End of feature extraction %%%

%%% SVM Classification %%%
% for idx = 1:size(featuresFileNames,1)
%     accuracyCellArray{idx}.vggSVMAccuracy = classifierSVM(mergedVggFileName(idx,:));
%     accuracyCellArray{idx}.imagenetSVMAccuracy = classifierSVM(mergedImagenetFileName(idx,:));
% end
% celldisp(accuracyCellArray)
%%% End of SVM Classification %%%