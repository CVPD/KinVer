function accuracy = classifierSVM(inputFile)

    feat = csvread(inputFile);

    cvIdx = length(feat);
    classIdx = cvIdx - 1;

    cvFolds = {};
    prevFold = 1;
    startPos = 1;
    for idx = 1:size(feat,1)
        if prevFold ~= feat(idx,cvIdx)
            cvFolds{prevFold}.start = startPos;
            cvFolds{prevFold}.end = idx-1;
            startPos = idx;
            prevFold = feat(idx,cvIdx);
        end
    end
    cvFolds{prevFold}.start = startPos;
    cvFolds{prevFold}.end = idx;

    allReal = [];
    allPredicted = [];
    for idx = 1:size(cvFolds,2)
        trainIdx = [1:cvFolds{idx}.start-1 cvFolds{idx}.end+1:size(feat,1)];
        testIdx = cvFolds{idx}.start:cvFolds{idx}.end;
        svmModel = fitcsvm( feat(trainIdx,1:classIdx-1), feat(trainIdx,classIdx) );
        [predicted,~] = predict(svmModel, feat(testIdx,1:classIdx-1));
        allReal = [allReal;feat(testIdx,classIdx)];
        allPredicted = [allPredicted;predicted];
    end

    C = confusionmat(allReal,allPredicted);
    TP = C(1,1);
    TN = C(2,2);
    total = size(allReal,1);
    accuracy = (TP+TN)/total;
end