% function v3 = mergeTwoVectors(v1,v2)
% 
% Merges two vectors into one
% Input: v1 and v2; vectors to be merged
% Output: v3 merged vector
function v3 = mergeTwoVectors(v1,v2)

dividend = abs(v1-v2);
divisor = v1+v2;
v3 = dividend;%./divisor; % element wise division
v3(isnan(v3)) = 0; % Replace all NaN by 0s

end