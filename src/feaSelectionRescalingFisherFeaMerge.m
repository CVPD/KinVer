% function fea = feaSelectionRescalingFisherFeaMerge(fea, idxa, idxb, fold, ...
%    matches, K ,numSelect)
%
% Selects the most relevant descriptors of the input features using fisher
% feature selection (supervised) with rescaling (substracting mean and
% dividing by standard deviation)
%
% Input: fea; cell array that contains all the features extracted for all the pairs individuals
% Input: idxa; pairs' parent indexes. The same row of this vector and idxb's form a pair
% Input: idxb; pairs' child indexes. The same row of this vector and idxa's form a pair
% Input: fold; vector that indicates how train and test data are prepared to split in folds
% Input: matches; class of the instances
% Input: K; number of features
% Input: numSelect; the number of descrptors that are selected for each feature
% Output: fea; input features after selecting the most relevant descriptors

function fea = feaSelectionRescalingFisherFeaMerge(fea, idxa, idxb, fold, ...
    matches, K ,numSelect)

un = unique(fold);
nfold = length(un);

for c = 1%:nfold
    trainMask = fold ~= c;
    testMask = fold == c;
    tr_idxa = idxa(trainMask);
    tr_idxb = idxb(trainMask);
    tr_matches = matches(trainMask);
    ts_idxa = idxa(testMask);
    ts_idxb = idxb(testMask);
    ts_matches = matches(testMask);
    
    for p = 1:K
        X = fea{p};
        tr_Xa = X(tr_idxa, :);                    % training data
        tr_Xb = X(tr_idxb, :);                    % training data
        tr_Xc = mergePairsInMatrix(tr_Xa, tr_Xb);
        
        ranking = rankingRescaleOrderFisher(tr_Xc,tr_matches);
        numF = size(tr_Xc,2);
        numSelectedFeats = round(numF*numSelect(p));
        X = X(:,ranking(1:numSelectedFeats));
        fea{p} = X;
        clear X;
    end
end

end

function ranking = rankingRescaleOrderFisher(X,labels)

% Center each row (pair) and divide by its standar deviation
Xnew = zscore(X')';

% Build Laplacian matrix N.N
N = size(Xnew,1);
numFea = size(Xnew,2);
numClass0 = sum(labels==0);
numClass1 = sum(labels==1);
coefClass0 = 1/numClass0;
coefClass1 = 1/numClass1;
for i = 1:N
    for j = 1:N
        if labels(i) == labels(j)
            if (labels(i) == 0)
                Sw(i,j) = coefClass0;
            else
                Sw(i,j) = coefClass1;
            end
        else
            Sw(i,j) = 0;
        end
    end
end

Lw = eye(N) - Sw;

% Create ranking of features
for i = 1:numFea
    fi = Xnew(:,i);
    s(i)=fi' * Lw * fi;
end

[~,~,ranking] = unique(s);
ranking = ranking';
end
