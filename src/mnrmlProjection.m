% function mnrmlSpaceChange(inputFile,outputFile)
%
% Transforms the input data arranged to perform classification to the MNRML
% created space.
% Input: inputFile; data ready to perform classification per fold
% Input: outputFile; The file name where the input data transformed to the new
% MNRML space will be stored.
%% Example of call to the function
% inputFile = 'C:\Users\oscar\Desktop\TFM\project\data\classification_data_ms.mat.mat';
% outputFile = strcat(inputFile(1:length(classificationDataFileName)-4),'_mnrml.mat');
% mnrmlSpaceChange(inputFile,outputFile);
function [projFea,W,beta] = mnrmlProjection(fea, idxa, idxb, fold, matches, K, T, knn, Wdims)

disp('mnrml projection started. Folds: ')
    
addpath('external/NRML/nrml');

un = unique(fold);
nfold = length(un);

%% NRML
t_sim = [];
t_ts_matches = [];
t_acc = zeros(nfold, 1);
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
    
    %% do PCA  on training data
    for p = 1:K
        X = fea{p};
        tr_Xa = X(tr_idxa, :);                    % training data
        tr_Xb = X(tr_idxb, :);                    % training data
        [eigvec, eigval, ~, sampleMean] = PCA([tr_Xa; tr_Xb]);
        Wdims = size(eigvec, 2);
        X = (bsxfun(@minus, X, sampleMean) * eigvec(:, 1:Wdims));
        
        N = size(X, 1);
        for i = 1:N
            X(i, :) = X(i, :) / norm(X(i, :));
        end
        tr_Xa_pos{p} = X(tr_idxa(tr_matches), :); % positive training data
        tr_Xb_pos{p} = X(tr_idxb(tr_matches), :); % positive training data
        ts_Xa{p} = X(ts_idxa, :);                 % testing data
        ts_Xb{p} = X(ts_idxb, :);                 % testing data
        feaPCA{p} = X;
        clear X;
    end
    %% MNRML
    [W{c}, beta{c}] = mnrml_train(tr_Xa_pos, tr_Xb_pos, knn, Wdims, T);
    
    for p = 1:K
        projFea{c}{p} = feaPCA{p} * W{c};
    end
    
    clear feaPCA; % Until here, OK (to test:   a=projFea{1}{1}(ts_idxa,:);  )
end

disp('mnrml projection finished')

end