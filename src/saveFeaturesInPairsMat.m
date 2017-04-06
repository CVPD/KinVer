function saveFeaturesInPairsMat(dataFile,metadataDir,VGGFileName,imagenetFileName)

load(dataFile);
load(metadataDir);

names = cell(length(data),1);
for idx = 1:length(data)
    names{idx} = data{idx}.name;
end

% Vgg features to matrix
for idx = 1:length(data)
    ux(idx,:) = data{idx}.vggFeat;
end

% Pairs to idxs
for idx = 1:length(pairs)
    % Find vectors of the pair
    vec1Idx = find(ismember(names,pairs(idx,3)));
    vec2Idx = find(ismember(names,pairs(idx,4)));
    if isempty(vec1Idx) || isempty(vec2Idx)
        error('One of the pair images is not found in the processed data');
    end
    idxa(idx,:) = vec1Idx;
    idxb(idx,:) = vec2Idx;
    numFoldCell = pairs(idx,1); % number of fold to perform CV
    fold(idx,:) = numFoldCell{1};
    realClassCell = pairs(idx,2); % real class
    matches(idx,:) = realClassCell{1};
end

matches = logical(matches); % turn it into boolean

save(VGGFileName, 'ux', 'matches', 'idxa', 'idxb', 'fold');

clear ux matches idxa idxb fold;

% Vgg features to matrix
for idx = 1:length(data)
        ux(idx,:) = data{idx}.imagenetFeat;
end

% Pairs to idxs
for idx = 1:length(pairs)
    % Find vectors of the pair
    vec1Idx = find(ismember(names,pairs(idx,3)));
    vec2Idx = find(ismember(names,pairs(idx,4)));
    if isempty(vec1Idx) || isempty(vec2Idx)
        error('One of the pair images is not found in the processed data');
    end
    idxa(idx,:) = vec1Idx;
    idxb(idx,:) = vec2Idx;
    numFoldCell = pairs(idx,1); % number of fold to perform CV
    fold(idx,:) = numFoldCell{1};
    realClassCell = pairs(idx,2); % real class
    matches(idx,:) = realClassCell{1};
end

matches = logical(matches); % turn it into boolean

save(imagenetFileName, 'ux', 'matches', 'idxa', 'idxb', 'fold');

clear ux matches idxa idxb fold;

end