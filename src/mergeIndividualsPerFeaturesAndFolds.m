% function ind3 = mergeIndividualsPerFeaturesAndFolds(ind1,ind2)
% 
% Merges two individuals (that form a pair) per features per each fold
% calling mergeTwoVectors function
% Input: ind1 and ind2; cell arrays that contain data of folds, features and individuals
% Output: ind3; the same structure as input but merging individuals in pairs
function ind3 = mergeIndividualsPerFeaturesAndFolds(ind1,ind2)

numFolds = size(ind1,2);
numFeat = size(ind1{1},1);

for fold = 1:numFolds
    numPairs = size(ind1{fold}{1},1);
    ind3{fold} = cell(numFeat,1);
    for feat = 1:numFeat
        for pairIdx = 1:numPairs
            vec1 = ind1{fold}{feat}(pairIdx,:);
            vec2 = ind2{fold}{feat}(pairIdx,:);
            vec3 = mergeTwoVectors(vec1,vec2);
            ind3{fold}{feat}(pairIdx,:) = vec3;
        end
    end
end

end