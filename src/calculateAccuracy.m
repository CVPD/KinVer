% function accuracy = calculateAccuracy(realClass,predictedClass)
% 
% Calculates the accuracy from a classifier given the real and predicted classes
% Input: realClass; a vector that contains the real class of the data used
% to test the classifier
% Input: predictedClass; a vector that contains the predicted class of the data
% used to test the classifier
% Output: accuracy; the accuracy of the classifier being tested
function accuracy = calculateAccuracy(realClass,predictedClass)

% If predicted class are probabilities, turn into class
predictedClass(predictedClass>0.5) = 1;
predictedClass(predictedClass<1) = 0;
realClass = double(realClass);
predictedClass = double(predictedClass);

% Calculate accuracy using confusion matrix
C = confusionmat(realClass,predictedClass);
TP = C(1,1);
TN = C(2,2);
total = size(realClass,1);
accuracy = (TP+TN)/total;

end