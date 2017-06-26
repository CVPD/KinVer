% Calling example
% load('varAccuracyMNRMLIdxallKinFaceII.mat');
% [fisherDim,wdims]=calculateFisherDim(accuracyMNRMLIdx);
function [selectedDim,pcaDim] = calculateFisherDim(accuracyMNRMLIdx)
rangeFisherDimOuter = [0.5 0.4 0.3 0.2 0.1 0.075 0.05 0.025];
rangeFisherDimInner = [0.5 0.4 0.3 0.2 0.1 0.075 0.05 0.025];
pcaDimChange=length(15:70);
pcaDimChange=pcaDimChange+1; %For 0s line
searchRange = length(rangeFisherDimInner);
for idx = 1:4
	[~,pos] = max(accuracyMNRMLIdx);
	div=pos(idx)/pcaDimChange;
	numOuter=floor(div/searchRange);
	numInner=floor(div-numOuter*searchRange);
	selectedDim(idx,:)=[rangeFisherDimOuter(numOuter+1) rangeFisherDimInner(numInner+1)];
	startPos=(numOuter*searchRange+numInner)*pcaDimChange+1;
	pcaChangeRange=accuracyMNRMLIdx(startPos:startPos+pcaDimChange-2,:);
	[~,pos] = max(pcaChangeRange);
	pos = pos + 14;
	pcaDim(idx)=pos(idx);
end
end
