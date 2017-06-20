function fea = feaSelectionFisherConcat(fea, idxa, idxb, fold, ...
    matches, K ,numSelect)

addpath('./external/fisher/lib'); % dependencies
addpath('./external/fisher/methods'); % FS methods

un = unique(fold);
nfold = length(un);

for c = 1%:nfold
    trainMask = fold ~= c;
    testMask = fold == c;
    tr_idxa = idxa(trainMask);
    tr_idxb = idxb(trainMask);
    tr_matches = matches(trainMask);
    ts_idxa = idxa(testMask);
    ts_idxb = idxb(testMask);
    ts_matches = matches(testMask);

    for p = 1:K
        X = fea{p};
        tr_Xa = X(tr_idxa, :);                    % training data
        tr_Xb = X(tr_idxb, :);                    % training data
        
        tr_Xc = [tr_Xa tr_Xb];
  
        feaSize = size(tr_Xa,2);
        numF = size(tr_Xc,2);
        tr_matches = tr_matches + 0; % turn into double
        tr_matches(tr_matches==0,:) = -1;
        ranking = spider_wrapper(tr_Xc,tr_matches,numF,'fisher');
        
        ranking_a = ranking(ranking<=feaSize);
        ranking_b = ranking(ranking>feaSize);
        
        numSelectedFeats = round(feaSize*numSelect);
        selectedFeats = [ranking_a(1:numSelectedFeats) ranking_b(1:numSelectedFeats)-feaSize];
        uniqueSelectedFeats = unique(selectedFeats);
        X = X(:,uniqueSelectedFeats);
        fea{p} = X;
        clear X;
    end
end

end