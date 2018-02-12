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

disp( ['selDim1: ' num2str( feaSelectionDims(1) ) ', selDim2: ' num2str( feaSelectionDims(2) ) ] );
disp( ['wdims: ' num2str(wdims) ] );

load(vggMatFileName, 'ux', 'idxa', 'idxb', 'fold', 'matches' );
for kk=1:size( ux,1 )
   f = ux( kk, : );
   ux(kk,:) = f / norm( f, 2 );
end
fea{1} = ux;
clear ux idxa idxb fold matches;
load(imagenetMatFileName, 'ux', 'idxa', 'idxb', 'fold', 'matches' );
for kk=1:size( ux,1 )
   f = ux( kk, : );
   ux(kk,:) = f / norm( f, 2 );
end
fea{2} = ux;
%clear ux idxa idxb fold matches;
%load(LBPMatFileName);
%fea{3} = ux;
%clear ux idxa idxb fold matches;
%load(HOGMatFileName);
%fea{4} = ux;
K = 2;
% Classification on original features
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
