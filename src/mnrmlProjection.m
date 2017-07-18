% function [projFea,W,beta] = mnrmlProjection(fea, idxa, idxb, fold, ...
%    matches, K, T, knn)
%
% Transforms the input data arranged to perform classification to the MNRML
% created space using all the input features.
%
% Input: fea; cell array that contains all the features extracted for all the pairs individuals
% Input: idxa; pairs' parent indexes. The same row of this vector and idxb's form a pair
% Input: idxb; pairs' child indexes. The same row of this vector and idxa's form a pair
% Input: fold; vector that indicates how train and test data are prepared to split in folds 
% Input: matches; class of the instances.
% Input: K; number of features.
% Input: T; Number of iterations for the MNRML projection.
% Input: knn; Number of neighbours that form a neighborhood for the MNRML projection.
% Output: projFea; original features (fea) projected to PCA and MNRML space.
% Output: W; projection matrix created by MNRML.
% Output: beta; Coeficient that shows the relevance of each feature. It consists on several values (one per feature and the sum of all must equal 1).
function [projFea,W,beta] = mnrmlProjection(fea, idxa, idxb, fold, ...
    matches, K, T, knn)

disp('MNRML projection started. Folds: ')

addpath('external/NRML/nrml');

un = unique(fold);
nfold = length(un);

%% NRML
Wdims = inf;
for p = 1:K
    newSize = size(fea{p},2);
        if newSize < Wdims
            Wdims = newSize;
        end
end
for c = 1:nfold
    
    % Display number of fold processing
    txt = strcat('fold number', num2str(c));
    disp(txt)
    disp('')
    
    trainMask = fold ~= c;
    testMask = fold == c;
    tr_idxa = idxa(trainMask);
    tr_idxb = idxb(trainMask);
    tr_matches = matches(trainMask);
    
    %% do PCA  on training data
    for p = 1:K
        X = fea{p};
        tr_Xa_pos{p} = X(tr_idxa(tr_matches), 1:Wdims); % positive training data
        tr_Xb_pos{p} = X(tr_idxb(tr_matches), 1:Wdims); % positive training data
        feaPCA{p} = X;
        clear X;
    end
    %% MNRML
    [W{c}, beta{c}] = mnrml_train(tr_Xa_pos, tr_Xb_pos, knn, Wdims, T);
    
    for p = 1:K
        projFea{c}{p} = feaPCA{p}(:,1:Wdims) * W{c};
    end
    
    clear feaPCA;
end

disp('MNRML projection finished')


end

% Returns the maximum value of Wdims (the value that holds percEigVal information
% of the feature that needs the biggest information) of all the folds
function maxWdims = calculateWdims(percEigVal, fea, idxa, idxb, fold, K)

un = unique(fold);
nfold = length(un);

maxWdims = 0;

for c = 1:nfold
    trainMask = fold ~= c;
    tr_idxa = idxa(trainMask);
    tr_idxb = idxb(trainMask);
    
    for p = 1:K
        X = fea{p};
        tr_Xa = X(tr_idxa, :);                    % training data
        tr_Xb = X(tr_idxb, :);                    % training data
        [~, eigval, ~, ~] = PCA([tr_Xa; tr_Xb]);
        totalEig = sum(eigval);
        accum = 0;
        idx = 0;
        while accum/totalEig < percEigVal
            idx = idx + 1;
            accum = accum + eigval(idx);
        end
        Wdims = idx;
        if Wdims > maxWdims
            maxWdims = Wdims;
        end
    end
end
end