% inputFile = 'C:\Users\oscar\Desktop\TFM\project\data\mnrmlFeat_ms.mat';
% inputFile = 'C:\Users\oscar\Desktop\TFM\project\data\mnrmlFeat_ms_noW.mat';
% SVMClassification(inputFile);

function [mergedFeaTr, mergedFeaTs] = convertEachPairIntoIndividual(fea, idxa, idxb, fold, K)

un = unique(fold);
nfold = length(un);

for c = 1:nfold
    trainMask = fold ~= c;
    testMask = fold == c;
    tr_idxa = idxa(trainMask);
    tr_idxb = idxb(trainMask);
    ts_idxa = idxa(testMask);
    ts_idxb = idxb(testMask);
    
    % Merge parent and child 2 feature vectors into one (Train data)
    for p = 1:K
        if iscell(fea{1}) % Features per fold
            X = fea{c}{p};
        else
            X = fea{p}; % Features (no folds)
        end
        tr_Xa = X(tr_idxa, :);                    % training data
        tr_Xb = X(tr_idxb, :);                    % training data
        mergedFeaTr{c}{p} = mergePairsInMatrix(tr_Xa, tr_Xb);
        
        ts_Xa = X(ts_idxa, :);                 % testing data
        ts_Xb = X(ts_idxb, :);                 % testing data
        mergedFeaTs{c}{p} = mergePairsInMatrix(ts_Xa, ts_Xb);
    end
end

end