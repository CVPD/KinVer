%% demo: MNRML

clear all;
clc;

addpath('nrml');

%% data & parametres
%load('data/LBP_KinFaceW-II_FS.mat');
load('C:\Users\oscar\Desktop\TFM\project\data\vgg_ms.mat');
fea{1} = ux;
clear ux matches idxa idxb fold;

%load('data/HOG_KinFaceW-II_FS.mat');
load('C:\Users\oscar\Desktop\TFM\project\data\imagenet_ms.mat');
fea{2} = ux;
%clear ux;

K = 2;        % the number of features
T = 1;        % Iterations
knn = 5;      % k-nearest neighbors
Wdims = 100;  % low dimension

un = unique(fold);
nfold = length(un);

numIt = 0;
rangeVal = 1:5:200;
for newVal = rangeVal
    strcat('------ new val: ', string(newVal), ' ------ ')
    Wdims = newVal;
    numIt = numIt+1;
    %% NRML
    t_sim = [];
    t_ts_matches = [];
    t_acc = zeros(nfold, 1);
    for c = 1:nfold
        trainMask = fold ~= c;
        testMask = fold == c;
        tr_idxa = idxa(trainMask);
        tr_idxb = idxb(trainMask);
        tr_matches = matches(trainMask);
        ts_idxa = idxa(testMask);
        ts_idxb = idxb(testMask);
        ts_matches = matches(testMask);
        
        %% do PCA  on training data
        for p = 1:K
            X = fea{p};
            tr_Xa = X(tr_idxa, :);                    % training data
            tr_Xb = X(tr_idxb, :);                    % training data
            [eigvec, eigval, ~, sampleMean] = PCA([tr_Xa; tr_Xb]);
            Wdims = size(eigvec, 2);
            X = (bsxfun(@minus, X, sampleMean) * eigvec(:, 1:Wdims));
            
            N = size(X, 1);
            for i = 1:N
                X(i, :) = X(i, :) / norm(X(i, :));
            end
            tr_Xa_pos{p} = X(tr_idxa(tr_matches), :); % positive training data
            tr_Xb_pos{p} = X(tr_idxb(tr_matches), :); % positive training data
            ts_Xa{p} = X(ts_idxa, :);                 % testing data
            ts_Xb{p} = X(ts_idxb, :);                 % testing data
            clear X;
        end
        %% MNRML
        [W, beta] = mnrml_train(tr_Xa_pos, tr_Xb_pos, knn, Wdims, T);
         
        %% cosine similarity
        sim = 0;
        for p = 1:K
            Xa = ts_Xa{p} * W;
            Xb = ts_Xb{p} * W;
            sim = sim + beta(p) * cos_sim(Xa', Xb');
        end
        t_sim = [t_sim; sim(:)];
        t_ts_matches = [t_ts_matches; ts_matches];
        
        %% Accuracy
        [~, ~, ~, ~, acc] = ROCcurve(sim, ts_matches);
        t_acc(c) = acc;
        fprintf('Fold %d, Accuracy = %6.4f \n', c, acc);
    end
    meanAcc(numIt,:) = mean(t_acc);
    fprintf('The mean accuracy = %6.4f\n', meanAcc(numIt,:));
end

plot(rangeVal,meanAcc);
title('Wdims range 1:5:200');

% %% plot ROC
% [fpr, tpr] = ROCcurve(t_sim, t_ts_matches);
% figure(1)
% plot(fpr, tpr);
% xlabel('False Positive Rate')
% ylabel('True Positive Rate')
% grid on;

%%