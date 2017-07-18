% function [mergedFeaTr, mergedFeaTs] = ldeProjection(mergedFeaTr, mergedFeaTs, fold, matches, K, K1, K2)
%
% Transforms the input data arranged to perform classification to the LDE
% created space.
%
% Input: mergedFeaTr; individuals training data (merged pairs features) ready to perform classification per fold
% Input: mergedFeaTs; individuals testing data (merged pairs features) ready to perform classification per fold
% Input: fold; vector that indicates how train and test data are prepared to split in folds 
% Input: matches; class of the instances
% Input: K; number of features
% Input: K1; LDE's parameter K1
% Input: K2; LDE's parameter K2
% Output: mergedFeaTr; individuals training data (merged pairs features ready to perform classification per fold. Projected to LDE space
% Output: mergedFeaTs; individuals testing data (merged pairs features) ready to perform classification per fold. Projected to LDE space
function [mergedFeaTr, mergedFeaTs] = ldeProjection(mergedFeaTr, mergedFeaTs, fold, matches, K, K1, K2)

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
    
    tr_matches = tr_matches';
    
    % Perform LDE projection
    for p = 1:K
        mergedFeaTr{c}{p} = transpose(mergedFeaTr{c}{p});
        mergedFeaTs{c}{p} = transpose(mergedFeaTs{c}{p});
        
        
        [vec val Ww Wb Lb Lw] = LDE_K1K2(mergedFeaTr{c}{p},tr_matches,K1,K2);
        mergedFeaTr{c}{p} = vec' * mergedFeaTr{c}{p};
        mergedFeaTs{c}{p} = vec' * mergedFeaTs{c}{p};
        
        mergedFeaTr{c}{p} = transpose(mergedFeaTr{c}{p});
        mergedFeaTs{c}{p} = transpose(mergedFeaTs{c}{p});
    end
end

disp('LDE projection finished')

end