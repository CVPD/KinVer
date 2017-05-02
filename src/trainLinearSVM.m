% function svmLinearModel = trainLinearSVM(trainingData,classData)
% 
% Trains a linear SVM classifier with the argument data
% Input: trainingData; a matrix that contains predictor variables (features)
% Input: classData; a vector that contains the class per each individual
% (row) in the trainingData matrix
% as columns and instances as rows
% Output: svmLinearModel; the created linear SVM classifier
function svmLinearModel = trainLinearSVM(trainingData,classData)

svmLinearModel = fitcsvm(trainingData, classData);

end