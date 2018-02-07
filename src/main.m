%% Initialization

%% Configuration setting
databaseID = 'KinFaceW-II';

% Define KinFaceW database path
dbDir='../datasets';
% Define matconvnet path
convnetDir = '../matconvnet';
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
rangeFisherDimOuter = [0.5 0.4 0.3 0.2 0.1 0.075 0.05 0.025];
rangeFisherDimInner = [0.5 0.4 0.3 0.2 0.1 0.075 0.05 0.025];
idx = 1;

rangeWdims = 15:70;%20:40;
totalIdxs = length( rangeFisherDimOuter ) * length( rangeFisherDimInner )...
    * length( rangeWdims );

for selDimFea1 = rangeFisherDimOuter
    for selDimFea2 = rangeFisherDimInner
        T = 4;
        knn = 6;

        perc = 0;
        %wdims =[59 47 60 47];% [26 43 39 37];
        
        for wdims = rangeWdims
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
            disp( ['***** idx = ' num2str( idx ) '/' num2str(totalIdxs) ' *****'] );
            tic
            for pairIdx = 1:size(featuresFileNames,1)
                [accuracy(pairIdx),accuracyMNRML(pairIdx),accuracyNRML(pairIdx), ...
                    accuracyPerFeat(pairIdx,:), numEigVals(pairIdx,idx),betaPerFeat(pairIdx,:)] = ...
                    performClassification(imagePairsDirs(pairIdx,:), convnetDir, ...
                    featuresFileNames(pairIdx,:), metadataPairs(pairIdx,:), ...
                    pairIdStrs(pairIdx,:), vggFaceFileNames(pairIdx,:), ...
                    vggFFileNames(pairIdx,:), LBPFileNames(pairIdx,:), ...
                    HOGFileNames(pairIdx,:), ...
                    T, knn, perc, K1, K2, wdims,sizeSVM,[selDimFea1 selDimFea2]);%wdims(pairIdx),sizeSVM);
                
            end
            toc
            meanAccuracy(idx) = mean(accuracyMNRML);
            betaMeans(idx,:) = mean(betaPerFeat,1);
            accuracyMNRMLIdx(idx,:) = accuracyMNRML;
            idx = idx+1;
        end
        tic
        save('experiment.mat', 'meanAccuracy', 'betaMeans', 'accuracyMNRMLIdx' );
        disp( 'Saved experiment.mat' );
        toc
    end
end
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