% function projFea = PCAplusNRMLprojections(fea, idxa, idxb, fold, matches, K, T, knn, Wdims)
%
% Transforms the input data arranged to perform classification to the NRML
% for all the input features individually.
%
% Input: fea; cell array that contains all the features extracted for all the pairs individuals
% Input: idxa; pairs' parent indexes. The same row of this vector and idxb's form a pair
% Input: idxb; pairs' child indexes. The same row of this vector and idxa's form a pair
% Input: fold; vector that indicates how train and test data are prepared to split in folds 
% Input: matches; class of the instances.
% Input: K; number of features.
% Input: T; Number of iterations for the MNRML projection.
% Input: knn; Number of neighbours that form a neighborhood for the MNRML projection.
% Input: Wdims; number of descriptors that are selected after performing PCA projection
% Output: projFea; original features (fea) projected to PCA and MNRML space.
function projFea = PCAplusNRMLprojections(fea, idxa, idxb, fold, matches, K, T, knn, Wdims)

disp('mnrml projection started. Folds: ')

addpath('external/NRML/nrml');

un = unique(fold);
nfold = length(un);

%% NRML
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
    ts_idxa = idxa(testMask);
    ts_idxb = idxb(testMask);
    ts_matches = matches(testMask);
    
    %% do PCA and NMRL on training data
    for p = 1:K
        X = fea{p};
        tr_Xa = X(tr_idxa, :);                    % training data
        tr_Xb = X(tr_idxb, :);                    % training data
        [eigvec, eigval, ~, sampleMean] = PCA([tr_Xa; tr_Xb], Wdims);
        Wdims = size(eigvec, 2);
        X = (bsxfun(@minus, X, sampleMean) * eigvec(:, 1:Wdims));
        
        N = size(X, 1);
        for i = 1:N
            X(i, :) = X(i, :) / norm(X(i, :));
        end
        tr_Xa_pos = X(tr_idxa(tr_matches), :); % positive training data
        tr_Xb_pos = X(tr_idxb(tr_matches), :); % positive training data
        ts_Xa = X(ts_idxa, :);                 % testing data
        ts_Xb = X(ts_idxb, :);                 % testing data
        
        %% NRML
        W = nrml_train(tr_Xa_pos, tr_Xb_pos, knn, Wdims, T);
        projFea{c}{p} = X*W;
        
        clear X;
    end
    
end

disp('mnrml projection finished')

end