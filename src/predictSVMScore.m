% function prediction = predictSVMScore(svmTrainedModel,predictData)
% 
% Uses a linear SVM to predict the score of the class 1 of the provided
% data.
% Input: svmModel; the trained model used to predict the class of predictData
% Input: predictData; a matrix that contains predictor variables (features)
% as columns and instances as rows
% Output: score; the score of belonging to the class 1 predicted for 
% predictData
function score = predictSVMScore(svmTrainedModel,predictData)

[~,score] = predict(svmTrainedModel, predictData);

score = score(:,2);

end