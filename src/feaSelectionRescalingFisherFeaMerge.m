function fea = feaSelectionFisherMerge(fea, idxa, idxb, fold, ...
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
