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
    txt = strcat('fold number', string(c));
    disp(txt)
    disp('')
    
    %% select dataset rows given indexes
    for p = 1:K
        Xa = Xtra{c}{p};
        Xb = Xtrb{c}{p};
        tr_Xa_pos{p} = Xa(tr_matches{c}, :); % positive training data
        tr_Xb_pos{p} = Xb(tr_matches{c}, :); % positive training data
        clear Xa Xb;
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