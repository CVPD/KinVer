function ranking = cfs(X)
% BASELINE - Sort features according to pairwise correlations

corrMatrix = abs( corr(X) );

% Ranking according to minimum correlations
scores = min(corrMatrix,[],2);


[~,ranking] = sort(scores,'ascend');

end