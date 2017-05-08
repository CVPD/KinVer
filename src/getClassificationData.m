% function getClassificationData(featFileNamesCell,destFileName)
%
% Gets the data ready to perform classification per fold.
% Input: featFileNamesCell; a cell array that contains the name (full path)
% of the files in which features are located. One name per cell.
% Input: destFileName; a string containing the full path of the file to
% store the new data (data ready to perform classification).
%% Example of call to the function
% featFileNamesCell{1} = 'C:\Users\oscar\Desktop\TFM\project\data\vgg_ms.mat';
% featFileNamesCell{2} = 'C:\Users\oscar\Desktop\TFM\project\data\imagenet_ms.mat';
% out = 'C:\Users\oscar\Desktop\TFM\project\data\classification_data_ms.mat';
% getClassificationData(featFileNamesCell, out);
function getClassificationData(featFileNamesCell,destFileName)

K = length(featFileNamesCell);

for idx = 1:length(featFileNamesCell)
    clear ux matches idxa idxb fold;
    load(featFileNamesCell{idx});
    fea{idx} = ux;
end

un = unique(fold);
nfold = length(un);

for c = 1:nfold
    trainMask = fold ~= c;
    testMask = fold == c;
    tr_idxa = idxa(trainMask);
    tr_idxb = idxb(trainMask);
    tr_matches{c} = matches(trainMask);
    ts_idxa = idxa(testMask);
    ts_idxb = idxb(testMask);
    ts_matches{c} = matches(testMask);
    
    %% select dataset rows given indexes
    for p = 1:K
        X = fea{p};
        
        tr_Xa_pos{p} = X(tr_idxa(tr_matches{c}), :); % positive training data
        tr_Xb_pos{p} = X(tr_idxb(tr_matches{c}), :); % positive training data
        tr_Xa{p} = X(tr_idxa, :);                 % training data
        tr_Xb{p} = X(tr_idxb, :);                 % training data
        ts_Xa{p} = X(ts_idxa, :);                 % testing data
        ts_Xb{p} = X(ts_idxb, :);                 % testing data
        clear X;
    end
    
    %% Get all the training and testing data of each feature per fold
    Xtra{c} = {};
    Xtrb{c} = {};
    Xtsa{c} = {};
    Xtsb{c} = {};
    for p = 1:K
        Xtra{c} = [Xtra{c}; tr_Xa{p}];
        Xtrb{c} = [Xtrb{c}; tr_Xb{p}];
        Xtsa{c} = [Xtsa{c}; ts_Xa{p}];
        Xtsb{c} = [Xtsb{c}; ts_Xb{p}];
    end
    
    % Assign the same importance (beta value) to each feature per fold
    for p = 1:K
        beta{c}(p) = 1/K;
    end
end

save(destFileName, 'Xtra', 'Xtrb', 'Xtsa', 'Xtsb', 'tr_matches', ...
    'ts_matches', 'beta');
end