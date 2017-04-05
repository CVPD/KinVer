function exportFeaturesToCSV(dataFile,metadataDir,mergedVGGFileName,mergedImagenetFileName)

load(dataFile);
load(metadataDir);

names = cell(length(data),1);
for idx = 1:length(data)
    names{idx} = data{idx}.name;
end

% vggFeatMerge calculus
vectorsLen = length(data{1}.vggFeat);
vggFeatMerged = zeros(length(pairs),vectorsLen+1);
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
    v2Res = reshape(vec2,[1,vectorsLen]);
    dividend = abs(v1Res-v2Res);
    divisor = v1Res+v2Res;
    vggFeatMerged(idx,1:vectorsLen) = dividend./divisor; % element wise division
    realClassCell = pairs(idx,2); % real class
    vggFeatMerged(idx,vectorsLen+1) = realClassCell{1};
    numFoldCell = pairs(idx,1); % number of fold to perform CV
    vggFeatMerged(idx,vectorsLen+2) = numFoldCell{1};
end

vggFeatMerged(isnan(vggFeatMerged)) = 0; % Replace all NaN by 0s

csvwrite(mergedVGGFileName,vggFeatMerged);

% imageNetFeatMerge calculus
vectorsLen = length(data{1}.imagenetFeat);
imagenetFeatMerged = zeros(length(pairs),vectorsLen+1);
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
    v2Res = reshape(vec2,[1,vectorsLen]);
    dividend = abs(v1Res-v2Res);
    divisor = v1Res+v2Res;
    imagenetFeatMerged(idx,1:vectorsLen) = dividend./divisor; % element wise division
    realClassCell = pairs(idx,2); % real class
    imagenetFeatMerged(idx,vectorsLen+1) = realClassCell{1};
    numFoldCell = pairs(idx,1); % number of fold to perform CV
    imagenetFeatMerged(idx,vectorsLen+2) = numFoldCell{1};
end
imagenetFeatMerged(isnan(imagenetFeatMerged)) = 0; % Replace all NaN by 0s
csvwrite(mergedImagenetFileName,imagenetFeatMerged);
end