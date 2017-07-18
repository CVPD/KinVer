function a = fisher(hyper)

%=========================================================================
% Select features based on Fisher/Correlation score  
%=========================================================================  
% A=FISHER(H) returns a fisher object initialized with hyperparameters H. 
%
% Calculates a Fisher/Correlation score for each feature to implement
%  feature selection.
%
% Hyperparameters, and their defaults
%  feat=[]              -- number of features
%  output_rank=1        -- set to 1 if only the feature ranking matters
%                          (does not perform any classification on the data)
%                          otherwise performs classification using
%                          weights given by individual correlation scores
%  method=2             -- useful only for multi-class. Set the how to combine
%                          the score of different one-vs-rest fisher's score. 
%                          (2 = take the sum, 1 = take the max)
% Model
%  w                    -- the weights
%  b0                   -- the threshold (when using all features)
%  rank                 -- the ranking of the features
%  d                    -- training set
%
% Methods:
%  train, test, get_w 
%
% Example:
% d=gen(toy); a=fisher; a.feat=10; a.output_rank=1;[r,a]=train(a,d);
% a.rank  % - lists the chosen features in  order of importance
%
% Note:
%  To use for Furey et al. method (e.g correlation coefficients + svm) 
%  use:  chain(fisher('output_rank=1'),svm)
%=========================================================================
% Reference : Neural Networks for Pattern Recognition
% Author    : C. Bishop
% Link      : http://www.amazon.com/exec/obidos/tg/detail/-/0198538642/qid=1080909371/sr=8-1/ref=pd_ka_1/002-6279399-2828812?v=glance&s=books&n=507846
%=========================================================================

  a.feat=[]; % number of features 
  a.output_rank=0; % don't output labels, output selected features 
  a.method=2;  
  % model           
  a.w=[];
  a.b0=0;
  a.rank=[];
  a.d=[];
  
  p=algorithm('fisher');
  a= class(a,'fisher',p);
  if nargin==1
    eval_hyper;
  end  
  
