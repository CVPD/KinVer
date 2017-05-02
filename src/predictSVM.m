% function prediction = predictSVM(svmTrainedModel,predictData)
% 
% Uses a linear SVM to predict the class of the provided data
% Input: svmModel; the trained model used to predict the class of predictData
% Input: predictData; a matrix that contains predictor variables (features)
% as columns and instances as rows
% Output: prediction; the class predicted for predictData
function prediction = predictSVM(svmTrainedModel,predictData)

[prediction,~] = predict(svmTrainedModel, predictData);

end