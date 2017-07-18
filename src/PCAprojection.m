% function projFea = PCAprojection(fea, idxa, idxb, fold, ...
%    matches, K, eigValPerc, wdims)
%
% Transforms the input data arranged to perform classification first with PCA projection and then dimension reduction by cropping to the wdims most important descriptors.
%
% Input: fea; cell array that contains all the features extracted for all the pairs individuals
% Input: idxa; pairs' parent indexes. The same row of this vector and idxb's form a pair
% Input: idxb; pairs' child indexes. The same row of this vector and idxa's form a pair
% Input: fold; vector that indicates how train and test data are prepared to split in folds 
% Input: matches; class of the instances.
% Input: K; number of features.
% Input: wdims and eigValPerc; number of descriptors that are selected after performing
    % PCA projection. If wdims equals to -1, then it will be calculated
    % automatically by mutiplying the size of the feature by eigValPerc factor.
% Output: projFea; original features (fea) projected to PCA and cropped.

function projFea = PCAprojection(fea, idxa, idxb, fold, ...
    matches, K, eigValPerc, wdims)

disp('PCA projection started. Folds: ')

addpath('external/NRML/nrml');

un = unique(fold);
nfold = length(un);

%% NRML
if wdims == -1
    Wdims = calculateWdims(eigValPerc, fea, idxa, idxb, fold, K);
else
    Wdims = wdims;
end
for c = 1:nfold
    
    % Display number of fold processing
    txt = strcat('fold number', num2str(c));
    disp(txt)
    disp('')
    
    trainMask = fold ~= c;
    tr_idxa = idxa(trainMask);
    tr_idxb = idxb(trainMask);
    
    %% do PCA  on training data
    for p = 1:K
        X = fea{p};
        tr_Xa = X(tr_idxa, :);                    % training data
        tr_Xb = X(tr_idxb, :);                    % training data
        [eigvec, ~, ~, sampleMean] = PCA([tr_Xa; tr_Xb]);
        X = (bsxfun(@minus, X, sampleMean) * eigvec(:, 1:Wdims));
        
        N = size(X, 1);
        for i = 1:N
            X(i, :) = X(i, :) / norm(X(i, :));
        end
        feaPCA{p} = X;
        clear X;
    end
    for p = 1:K
        projFea{c}{p} = feaPCA{p};
    end
end

disp('PCA projection finished')

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