% function svmGaussianModel = trainGaussianSVM(trainingData,classData)
% 
% Trains a gaussian SVM classifier with the argument data
% Input: trainingData; a matrix that contains predictor variables (features)
% Input: classData; a vector that contains the class per each individual
% (row) in the trainingData matrix
% as columns and instances as rows
% Output: svmGaussianModel; the created gaussian SVM classifier
function svmGaussianModel = trainGaussianSVM(trainingData,classData)

svmGaussianModel = fitcsvm(trainingData, classData, ...
    'KernelFunction','RBF','KernelScale','auto');

end