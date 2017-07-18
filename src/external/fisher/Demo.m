%  Matlab Code-Library for Feature Selection
%  A collection of S-o-A feature selection methods
%  Version 4.2.17 April 2017
%  Support: Giorgio Roffo
%  E-mail: giorgio.roffo@glasgow.ac.uk
%
%  Before using the Code-Library, please read the Release Agreement carefully.
%
%  Release Agreement:
%
%  - All technical papers, documents and reports which use the Code-Library will acknowledge the use of the library as follows: 
%    “The research in this paper use the Feature Selection Code Library (FSLib)” and a citation to:
%  
%  ------------------------------------------------------------------------
% If you use this toolbox (or method included in it), please cite:

%@article{roffo2017ranking,
%  title={Ranking to Learn: Feature Ranking and Selection via Eigenvector Centrality},
%  author={Roffo, Giorgio and Melzi, Simone},
%  journal={New Frontiers in Mining Complex Patterns, Fifth International workshop, nfMCP2016},
%  publisher={Springer - Lecture Notes in Computer Science},
%  year={2017}
%}

% @INPROCEEDINGS{Roffo7410835, 
% author={G. Roffo and S. Melzi and M. Cristani}, 
% booktitle={2015 IEEE International Conference on Computer Vision (ICCV)}, 
% title={Infinite Feature Selection}, 
% year={2015}, 
% pages={4202-4210}, 
% doi={10.1109/ICCV.2015.478}, 
% month={Dec},}
%  ------------------------------------------------------------------------

% Important:
% Before using the toolbox compile the solution:
% Please note, you need a compiler on your platform already installed. The compiler is not part of this toolbox,
% If you use Windows machines this can be added by installing MS Visual Studio 2015 (free from Microsoft).
% If you use other platform you need to instal a valid c++ compiler x86 or x64 according to your architecture.
% Note that / and \ in the path change along with the platform (Linux, Windows, MAC), so please open the make file and check before running it.
% Then just run the make file below.
% make;


%% DEMO FILE
clear all
close all
clc;
fprintf('\nFEATURE SELECTION TOOLBOX v 4.0 2016 - For Matlab \n');
% Include dependencies
addpath('./lib'); % dependencies
addpath('./methods'); % FS methods

% Select a feature selection method from the list
listFS = {'InfFS','ECFS','mrmr','relieff','mutinffs','fsv','laplacian','mcfs','rfe','L0','fisher','UDFS','llcfs','cfs'};

[ methodID ] = readInput( listFS );
selection_method = listFS{methodID}; % Selected

% Load the data and select features for classification
load fisheriris
X = meas; clear meas
% Extract the Setosa class
Y = nominal(ismember(species,'setosa')); clear species

% Randomly partitions observations into a training set and a test
% set using stratified holdout
P = cvpartition(Y,'Holdout',0.20);

X_train = double( X(P.training,:) );
Y_train = (double( Y(P.training) )-1)*2-1; % labels: neg_class -1, pos_class +1

X_test = double( X(P.test,:) );
Y_test = (double( Y(P.test) )-1)*2-1; % labels: neg_class -1, pos_class +1

% number of features
numF = size(X_train,2);

% feature Selection on training data
switch lower(selection_method)
    case 'mrmr'
        ranking = mRMR(X_train, Y_train, numF);
        
    case 'relieff'
        [ranking, w] = reliefF( X_train, Y_train, 20);
        
    case 'mutinffs'
        [ ranking , w] = mutInfFS( X_train, Y_train, numF );
        
    case 'fsv'
        [ ranking , w] = fsvFS( X_train, Y_train, numF );
        
    case 'laplacian'
        W = dist(X_train');
        W = -W./max(max(W)); % it's a similarity
        [lscores] = LaplacianScore(X_train, W);
        [junk, ranking] = sort(-lscores);
        
    case 'mcfs'
        % MCFS: Unsupervised Feature Selection for Multi-Cluster Data
        options = [];
        options.k = 5; %For unsupervised feature selection, you should tune
        %this parameter k, the default k is 5.
        options.nUseEigenfunction = 4;  %You should tune this parameter.
        [FeaIndex,~] = MCFS_p(X_train,numF,options);
        ranking = FeaIndex{1};
        
    case 'rfe'
        ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
        
    case 'l0'
        ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
        
    case 'fisher'
        ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
        
    case 'inffs'
        % Infinite Feature Selection 2015 updated 2016
        alpha = 0.5;    % default, it should be cross-validated.
        sup = 1;        % Supervised or Not
        [ranking, w] = infFS( X_train , Y_train, alpha , sup , 0 );    
        
    case 'ecfs'
        % Features Selection via Eigenvector Centrality 2016
        alpha = 0.5; % default, it should be cross-validated.
        ranking = ECFS( X_train, Y_train, alpha )  ;
        
    case 'udfs'
        % Regularized Discriminative Feature Selection for Unsupervised Learning
        nClass = 2;
        ranking = UDFS(X_train , nClass ); 
        
    case 'cfs'
        % BASELINE - Sort features according to pairwise correlations
        ranking = cfs(X_train);     
        
    case 'llcfs'   
        % Feature Selection and Kernel Learning for Local Learning-Based Clustering
        ranking = llcfs( X_train );
        
    otherwise
        disp('Unknown method.')
end

k = 2; % select the first 2 features

% Use a linear support vector machine classifier
svmStruct = svmtrain(X_train(:,ranking(1:k)),Y_train,'showplot',true);
C = svmclassify(svmStruct,X_test(:,ranking(1:k)),'showplot',true);
err_rate = sum(Y_test~= C)/P.TestSize; % mis-classification rate
conMat = confusionmat(Y_test,C); % the confusion matrix

fprintf('\nMethod %s (Linear-SVMs): Accuracy: %.2f%%, Error-Rate: %.2f \n',selection_method,100*(1-err_rate),err_rate);



% MathWorks Licence
% Copyright (c) 2016-2017, Giorgio Roffo
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
% 
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in
%       the documentation and/or other materials provided with the distribution
%     * Neither the name of the University of Verona nor the names
%       of its contributors may be used to endorse or promote products derived
%       from this software without specific prior written permission.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.
