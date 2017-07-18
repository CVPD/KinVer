function dat =  testing(alg,dat)
    
 feat=alg.feat; 
 if isempty(feat) 
     feat=length(alg.rank);   
 end;
 rank=alg.rank(1:feat);             % features to select
 dat=get(dat,[],rank);  % perform the feature selection
 dat=set_name(dat,[get_name(dat) ' -> ' get_name(alg)]);
  
 if alg.output_rank==0  %% output selected features, not label estimates
   dat=test(alg.child,dat);  % train underlying algorithm 
 end 
 
   
 
