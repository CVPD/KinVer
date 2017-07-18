function a = l0(c,hyper) 

%=========================================================================
% L0 zero-norm minimization (Weston,Elisseeff dual method)  object
%=========================================================================   
% A=L0(H) returns a l0 object initialized with hyperparameters H. 
%
% Minimizes the zero norm of the weight vector subject to
% separability constraints. Although it is possible to deal with the 
% non-seperable case (see paper), this is not currently implemented.
% 
%  Hyperparameters, and their defaults
%
%   feat=0         -- number of features to be selected, if feat=0, 
%                     then the minimum number of feature will be computed.
%   output_rank=1  -- whether a ranking is desired, if set to 0 then a
%                     classification is perfomed after feature selection.
%   find_min=1     -- stop with the minimum number of features if set to 1 
%                     otherwise stop as soon as reach feat nonzero features
%   child=svm      -- Set the classifiers to be used at each step 
%
%  Model
%
%  a.rank=[]       -- ranking of features
%  a.child=svm     -- base classifier trained at end of process
%
%  Methods:
%   train, test
%
%  Example:
%  d=gen(toy); a=l0; a.feat=20; a.output_rank=1;[r,a]=train(a,d);
%  a.rank  % - lists the chosen features in  order of importance, using 20 features
%
%  d=gen(toy); a=l0; a.feat=0; a.output_rank=1;[r,a]=train(a,d);
%  a.rank  % - lists the chosen features in  order of importance, using minimum number of features
%
%=========================================================================
% Reference : Use of the zero­norm with linear models and kernel methods
% Author    : J. Weston, A. Elisseeff, B. Schölkopf, and M. Tipping
% Link      : http://www.ai.mit.edu/projects/jmlr//papers/volume3/weston03a/weston03a.pdf
%=========================================================================

  a.feat=0;       % number of features 
  a.output_rank=1;
  a.find_min=1;
  % model
  a.rank=[];
  if nargin==0
    a.child=svm;  
  else 
    a.child=c; %% algorithm to use  
  end 
  p=algorithm('l0');
  a= class(a,'l0',p);
  if nargin==2
    eval_hyper;
  end

   
  
