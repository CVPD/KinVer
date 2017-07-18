function [W, beta] = mnrml_train(Xa, Xb, Knn, dim, T)
%	Description
%   [W] = MNRML_TRAIN(X,matches,K) 
%   is the training part of NRML algorithm
%
%	Inputs,
% 		Xa and Xb: data instance matching pair (K x N x d)
%		Knn:  Knn nearest neighbors
%       dim: Number of dimensions for W
%       T:  Number of Iterations
%
%   Outputs,
%       W: Distance projection matrix (d x dim)
%
%	
%   Copyright: Advanced Digital Science Center

N = size(Xa{1},1);
K = size(Xa,2);
beta = repmat(1/K, 1, K);
W = eye(size(Xa{1},2), size(Xa{1},2));
q = 2;

for r = 1:T
    for p = 1:K
        % get p-th view
        Xaa = Xa{p} * W;
        Xbb = Xb{p} * W;
        
        % search for K-nearest neighbors for Xa
        DtrueTraining = distMat(Xaa);
        [~, I] = sort(DtrueTraining, 2);
        Xa_knn_idx = I(:,2:Knn+1);
        
        % search for K-nearest neighbors for Xb
        DtrueTraining = distMat(Xbb);
        [~, I] = sort(DtrueTraining, 2);
        Xb_knn_idx = I(:,2:Knn+1);
        
        % solve for H1,H2 and H3
        H1 = 0;
        H2 = 0;
        H3 = 0;
        for i = 1:N
            for t1 = 1:Knn
                diff = Xaa(i,:) - Xbb(Xb_knn_idx(i,t1),:);
                H1 = H1 + diff'*diff;
                
                diff = Xaa(Xa_knn_idx(i,t1),:) - Xbb(i,:);
                H2 = H2 + diff'*diff;
            end
            diff = Xaa(i,:) - Xbb(i,:);
            H3 = H3 + diff'*diff;
        end
        H1n{p} = H1 / (Knn*N);
        H2n{p} = H2 / (Knn*N);
        H3n{p} = H3 / N;
    end
    
    % solve for new beta
    for p = 1:K
        beta(p) = (1/trace(W'*(H1n{p}+H2n{p}-H3n{p})*W))^(1/(q-1));
    end
    beta = beta / sum(beta);
    
    % solve for W
    H = 0;
    for p = 1:K
        H = H + beta(p)*(W'*(H1n{p}+H2n{p}-H3n{p})*W);
    end
    [Wtmp, eigval] = eig(H);
    temp = diag(eigval);
    [~, I] = sort(temp, 'descend');
    W = Wtmp(:, I);
end
W = W(:, 1:dim);