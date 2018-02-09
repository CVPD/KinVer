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

numFeats = 0;
if useVGGFace
    clear ux idxa idxb fold matches;
    load( vggMatFileName, 'ux', 'idxa', 'idxb', 'fold', 'matches' );
    for kk=1:size( ux,1 )
        f = ux( kk, : );
        ux(kk,:) = f / norm( f, 2 );
    end
    fea{numFeats+1} = ux;
    numFeats = numFeats + 1;
end
if useVGGF
    clear ux idxa idxb fold matches;
    load( imagenetMatFileName, 'ux', 'idxa', 'idxb', 'fold', 'matches' );
    for kk=1:size( ux,1 )
        f = ux( kk, : );
        ux(kk,:) = f / norm( f, 2 );
    end
    fea{numFeats+1} = ux;
    numFeats = numFeats + 1;
end
if useLBP
    clear ux idxa idxb fold matches;
    load( LBPMatFileName, 'ux', 'idxa', 'idxb', 'fold', 'matches' );
    fea{numFeats+1} = ux;
    numFeats = numFeats + 1;
end
if useHOG
    clear ux idxa idxb fold matches;
    load( HOGMatFileName, 'ux', 'idxa', 'idxb', 'fold', 'matches' );
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
        matches, K, T, knn);
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
