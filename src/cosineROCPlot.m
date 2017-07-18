% function cosineROCPlot(dataFile,metadataDir,pairIdStr)
% 
% Plots a ROC curve for each feature and each kinship relationship
%
% Input: dataFile; Origin file name that contains all the features for a kinship relationship
% Input: metadataDir; metadata directory for that kinship relationship
% Input: pairIdStr; the current kinship relationship's pair ID
function cosineROCPlot(dataFile,metadataDir,pairIdStr)

load(dataFile);

load(metadataDir);

names = cell(length(data),1);
for idx = 1:length(data)
    names{idx} = data{idx}.name;
end

% cosVggFeat as 5th column of pairs
vectorsLen = length(data{1}.vggFeat);
for idx = 1:length(pairs)
    % Find vectors of the pair
    vec1Idx = find(ismember(names,pairs(idx,3)));
    vec2Idx = find(ismember(names,pairs(idx,4)));
    if isempty(vec1Idx) || isempty(vec2Idx)
        error('One of the pair images is not found in the processed data');
    end
    vec1 = data{vec1Idx}.vggFeat;
    vec2 = data{vec2Idx}.vggFeat;
    v1Res = reshape(vec1,[1,vectorsLen]);
    coef1 = v1Res/norm(v1Res');
    v2Res = reshape(vec2,[1,vectorsLen]);
    coef2 = v2Res/norm(v2Res');
    pairs(idx,5) =  {dot(coef1,coef2)}; % dot product of vectors
end

% cosVggFeat as 6th column of pairs
vectorsLen = length(data{1}.imagenetFeat);
for idx = 1:length(pairs)
    % Find vectors of the pair
    vec1Idx = find(ismember(names,pairs(idx,3)));
    vec2Idx = find(ismember(names,pairs(idx,4)));
    if isempty(vec1Idx) || isempty(vec2Idx)
        error('One of the pair images is not found in the processed data');
    end
    vec1 = data{vec1Idx}.imagenetFeat;
    vec2 = data{vec2Idx}.imagenetFeat;
    v1Res = reshape(vec1,[1,vectorsLen]);
    coef1 = v1Res/norm(v1Res');
    v2Res = reshape(vec2,[1,vectorsLen]);
    coef2 = v2Res/norm(v2Res');
    pairs(idx,6) =  {dot(coef1,coef2)}; % dot product of vectors
end

% ROC cosVggFeat (5th column)
iterationVals = 0:0.05:1;
results =  zeros(length(iterationVals),5);
it = 1;
for threshold = iterationVals
    P = 0;
    N = 0;
    TP = 0;
    TN = 0;
    for idx = 1:length(pairs)
        curClass = 0;
        valueCell = pairs(idx,5);
        value = valueCell{1};
        if value >= threshold
            curClass = 1;
        end
        realClassCell = pairs(idx,2);
        realClass = realClassCell{1};
        if realClass
            P = P + 1;
        else
            N = N + 1;
        end
        if curClass == realClass
            if realClass
                TP = TP + 1;
            else
                TN = TN + 1;
            end
        end
    end
    % save threshold and data to create confusion matrix in results
    results(it,1) = threshold;
    results(it,2) = P;
    results(it,3) = N;
    results(it,4) = TP;
    results(it,5) = TN;
    it = it + 1;
end
P = results(:,2);
N = results(:,3);
TP = results(:,4);
TN = results(:,5);
FP = N-TN;
FN = P-TP;
TPR = TP./(TP+FN);
FPR = FP./(FP+TN);
figure;
plot(FPR,TPR);
hold on;

% ROC cosImageNet (6th column)
iterationVals = 0:0.05:1;
results =  zeros(length(iterationVals),6);
it = 1;
for threshold = iterationVals
    P = 0;
    N = 0;
    TP = 0;
    TN = 0;
    for idx = 1:length(pairs)
        curClass = 0;
        valueCell = pairs(idx,6);
        value = valueCell{1};
        if value >= threshold
            curClass = 1;
        end
        realClassCell = pairs(idx,2);
        realClass = realClassCell{1};
        if realClass
            P = P + 1;
        else
            N = N + 1;
        end
        if curClass == realClass
            if realClass
                TP = TP + 1;
            else
                TN = TN + 1;
            end
        end
    end
    % save threshold and data to create confusion matrix in results
    results(it,1) = threshold;
    results(it,2) = P;
    results(it,3) = N;
    results(it,4) = TP;
    results(it,5) = TN;
    it = it + 1;
end
P = results(:,2);
N = results(:,3);
TP = results(:,4);
TN = results(:,5);
FP = N-TN;
FN = P-TP;
TPR = TP./(TP+FN);
FPR = FP./(FP+TN);
plot(FPR,TPR);
legend('VGG features','ImageNet features');
title(strcat('Cos ROC ',pairIdStr));
end