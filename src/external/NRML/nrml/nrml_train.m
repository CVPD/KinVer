function W = nrml_train(Xa, Xb, K, dim, T)
%	Description
%   [W] = NRML_TRAIN(X,matches,K) 
%   is the training part of NRML algorithm
%
%	Inputs,
% 		Xa and Xb: data instance matching pair (N x d)
%		K:  K nearest neighbors
%       dim: Number of dimensions for W
%       T:  Number of Iterations
%
%   Outputs,
%       W: Distance projection matrix (d x dim)
%
%	
%   Copyright: Advanced Digital Science Center

N = size(Xa,1);

for r = 1:T
    % search for K-nearest neighbors for Xa
    DtrueTraining = distMat(Xa);
    [~, I] = sort(DtrueTraining, 2);
    Xa_knn_idx = I(:,2:K+1);
    % search for K-nearest neighbors for Xb
    DtrueTraining = distMat(Xb);
    [~, I] = sort(DtrueTraining, 2);
    Xb_knn_idx = I(:,2:K+1);
    
    % solve for H1,H2 and H3
    H1 = 0;
    H2 = 0;
    H3 = 0;
    for i=1:N
        for t1 = 1:K
            diff = Xa(i,:) - Xb(Xb_knn_idx(i,t1),:);
            H1 = H1 + diff'*diff;
            
            diff = Xa(Xa_knn_idx(i,t1),:) - Xb(i,:);
            H2 = H2 + diff'*diff;
        end
        diff = Xa(i,:) - Xb(i,:);
        H3 = H3 + diff'*diff;
    end
    H1 = H1 / (K*N);
    H2 = H2 / (K*N);
    H3 = H3 / N;
    
    % eigendecomposition
    [Wtmp, eigval] = eig(H1+H2-H3);
    temp = diag(eigval);
    [~, I] = sort(temp, 'descend');
    W = Wtmp(:, I);
    
    % update Xa and Xb using W
    Xa = Xa * W;
    Xb = Xb * W;
end
W = W(:, 1:dim);