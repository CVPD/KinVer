function ranked = spider_wrapper( X_train,Y_train,numF,method )
% Wrapper for Spider toolbox
% It's a library of objects in Matlab
% It is meant to handle (reasonably) large unsupervised, supervised or semi-supervised machine learning problems.
fprintf('\n+ Feature selection method: %s \n',method);

dset = data((X_train),Y_train);
eval(['a=',method,';']);
a.feat=numF;
a.method='classification';
a.output_rank=1;
[tr,a]=train(a,dset);
ranked = a.rank;



% Other info: http://people.kyb.tuebingen.mpg.de/spider/download_frames.html