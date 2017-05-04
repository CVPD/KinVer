% function svmLinearModel = trainLinearSVMProbability(trainingData,classData)
% 
% Trains a linear SVM classifier with the argument data that predicts the
% probability of each instance belonging to class 1
% Input: trainingData; a matrix that contains predictor variables (features)
% Input: classData; a vector that contains the probability of each instance
% to belong to the class 1
% Output: svmLinearModel; the created linear SVM probability classifier
function svmLinearModel = trainLinearSVMProbability(trainingData,classData)

svmLinearModel = fitcsvm(trainingData, classData);%, 'Standardize',true);
%svmLinearModel = fitPosterior(svmLinearModel);

end