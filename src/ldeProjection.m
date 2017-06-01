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
function [mergedFeaTr, mergedFeaTs] = ldeProjection(mergedFeaTr, mergedFeaTs, fold, matches, K)

disp('LDE projection started. Folds: ')

addpath('external');


un = unique(fold);
nfold = length(un);

for c = 1:nfold
    % Display number of fold processing
    txt = strcat('fold number', num2str(c));
    disp(txt)
    disp('')
    
    trainMask = fold ~= c;
    testMask = fold == c;
    tr_matches = matches(trainMask);
    
    % Perform LDE projection
    for p = 1:K
        K1 = 4;
        K2 = 6;
        mergedFeaTr{c}{p} = transpose(mergedFeaTr{c}{p});
        mergedFeaTs{c}{p} = transpose(mergedFeaTs{c}{p});
        tr_matches = tr_matches';
        
        [vec val Ww Wb Lb Lw] = LDE_K1K2(mergedFeaTr{c}{p},tr_matches,K1,K2);
        mergedFeaTr{c}{p} = vec' * mergedFeaTr{c}{p};
        mergedFeaTs{c}{p} = vec' * mergedFeaTs{c}{p};
        
        mergedFeaTr{c}{p} = transpose(mergedFeaTr{c}{p});
        mergedFeaTs{c}{p} = transpose(mergedFeaTs{c}{p});
    end
end

disp('LDE projection finished')

end