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
function mnrmlSpaceChange(inputFile,outputFile)

load(inputFile);

addpath('external/NRML/nrml');


T = 1;        % Iterations
knn = 5;      % k-nearest neighbors
Wdims = 100;  % low dimension


%% NRML
nfold = size(Xtra,2);
K = size(Xtra{1},1);

for c = 1:nfold
    
    % Display number of fold processing
    txt = strcat('fold number', num2str(c));
    disp(txt)
    disp('')
    
    %% select dataset rows given indexes
    for p = 1:K
        tr_Xa_pos{p} = Xtra{c}{p}(tr_matches{c}, :); % positive training data
        tr_Xb_pos{p} = Xtrb{c}{p}(tr_matches{c}, :); % positive training data
    end
    
    %% do PCA  on training data (comment this for loop to remove PCA)
    for p = 1:K
        % Calculate eigen vectors and values
        [eigvec, eigval, ~, sampleMean] = PCA([Xtra{c}{p}; Xtrb{c}{p}], Wdims);
        Wdims = size(eigvec, 2); % Calculate dimension of matrix
        
        % Apply PCA to all data
        Xtra{c}{p} = applyPCA(Xtra{c}{p},eigvec,sampleMean,Wdims);
        Xtrb{c}{p} = applyPCA(Xtrb{c}{p},eigvec,sampleMean,Wdims);
        
        Xtsa{c}{p} = applyPCA(Xtsa{c}{p},eigvec,sampleMean,Wdims);
        Xtsb{c}{p} = applyPCA(Xtsb{c}{p},eigvec,sampleMean,Wdims);
        
        tr_Xa_pos{p} = applyPCA(tr_Xa_pos{p},eigvec,sampleMean,Wdims);
        tr_Xb_pos{p} = applyPCA(tr_Xb_pos{p},eigvec,sampleMean,Wdims);
        
    end
    
    %% MNRML
    [W, beta{c}] = mnrml_train(tr_Xa_pos, tr_Xb_pos, knn, Wdims, T);
    
    %% Get all the training and testing data of each feature per fold
    for p = 1:K
        Xtra{c}{p} = Xtra{c}{p} * W;
        Xtrb{c}{p} = Xtrb{c}{p} * W;
        Xtsa{c}{p} = Xtsa{c}{p} * W;
        Xtsb{c}{p} = Xtsb{c}{p} * W;
    end
end

save(outputFile, 'Xtra', 'Xtrb', 'Xtsa', 'Xtsb', 'tr_matches', 'ts_matches', 'beta');
end

% Input: X; Data to perform PCA
% Output: X; PCA performed in the input data
function X = applyPCA(X,eigvec,sampleMean,Wdims)

X = (bsxfun(@minus, X, sampleMean) * eigvec(:, 1:Wdims));

N = size(X, 1);
for i = 1:N
    X(i, :) = X(i, :) / norm(X(i, :));
end

end