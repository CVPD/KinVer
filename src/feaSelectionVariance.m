% Unsupervised feature selection
function selectedFea = feaSelection(fea, K)

for p = 1:K
    data = fea{p};
    if p<3
        variance = var(data);
        selected = variance>0;
        selectedFea{p} = data(:,selected);
    else
        selectedFea{p} = fea{p};
end
end
