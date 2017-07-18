% indM3 = mergePairsInMatrix(indM1, indM2)
% 
% Merges two individual matrixes that represent pairs into one matrix
%
% Input: indM1 and indM2; individual matrixes to be merged
% Output: indM3 pairs merged into matrix

function indM3 = mergePairsInMatrix(indM1, indM2)

indM3 = zeros(size(indM1,1), size(indM1,2));
for idx = 1:size(indM1,1)
    indM3(idx,:) = mergeTwoVectors(indM1(idx,:),indM2(idx,:));
end

end