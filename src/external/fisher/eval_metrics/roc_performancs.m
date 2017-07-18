function [ prec, recall, fpr, AUCroc, Acc, F1_accuracy, F1_Precision, F1_Recall ] = roc_performancs( labels, scores , plot_flag)
%  Matlab Code-Library for Feature Selection
%  A collection of S-o-A feature selection methods
%  Version 4.0 August 2016
%  Support: Giorgio Roffo
%  E-mail: giorgio.roffo@univr.it
%
%  Before using the Code-Library, please read the Release Agreement carefully.
%
%  Release Agreement:
%
%  - All technical papers, documents and reports which use the Code-Library will acknowledge the use of the library as follows: 
%    “The research in this paper use the Feature Selection Code Library (FSLib)” and a citation to:
%  
%  ------------------------------------------------------------------------
% @ARTICLE {roffoFSLib16, 
%     author = "Giorgio Roffo", 
%     title = "Feature Selection Techniques for Classification: A widely applicable code library", 
%     journal = "arXiv:1607.01327 [cs.CV]", 
%     year = "2016", 
%     month = "aug", 
%     note = "Thu, 18 Aug 2016 12:07:43 GMT" 
% }
%  ------------------------------------------------------------------------

% EXAMPLE
% Call the function 
% roc_performancs( [1 1 1 1 -1 -1 -1 -1]', [0.2 .8 .1 .3 -.1 -.7 0.01 -0.05]' , 1)


[Xfpr,Ytpr,~,AUCroc]  = perfcurve(double(labels), double(scores), 1,'TVals','all','xCrit', 'fpr', 'yCrit', 'tpr');
[Xpr,Ypr,~,AUCpr] = perfcurve(double(labels), double(scores), 1, 'TVals','all','xCrit', 'reca', 'yCrit', 'prec');
[acc,~,~,~] = perfcurve(double(labels), double(scores), 1,'xCrit', 'accu');

prec = Ypr; prec(isnan(prec))=1;
tpr = Ytpr,1; tpr(isnan(tpr))=0;% recall = true positive rate
fpr = Xfpr; % (1 - Specificity)
recall = tpr;

% Compute F-Measure
f1= 2*(prec.*tpr) ./ (prec+tpr);
[Max_F1,idx] = max(f1);
F1_Precision = prec(idx);
F1_tRecall = tpr(idx);
F1_accuracy = acc(idx);

if plot_flag
    figure;
    subplot(1,2,1)
    plot([tpr], [ prec], '-b', 'linewidth',2); % add pseudo point to complete curve
    xlabel('recall');
    ylabel('precision');
    grid on
    title(['precision-recall ']);
    
    subplot(1,2,2)
    plot([fpr], [tpr], '-r', 'linewidth',2); % add pseudo point to complete curve
    xlabel('false positive rate');
    ylabel('true positive rate');
    grid on
    title(['ROC curve']);
end

AUCroc = 100*AUCroc; % Area Under the ROC curve
Acc = 100*sum(labels == sign(scores))/length(scores); % Accuracy


end

