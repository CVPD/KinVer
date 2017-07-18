function [res,retAlg] =  training(alg,dat)
% [results,algorithm] =  train(algorithm,data,trn)
dorig=dat;
[numEx,vDim,oDim]= get_dim(dat);

disp(['training ' get_name(alg) '.... '])



%% if we use the previous trainings
if alg.algorithm.use_prev_train==1&alg.algorithm.trained==1,
    dat=get(dat,[],alg.rank(1:alg.feat));
else
    feat=alg.feat; % number of features desired
    left=[1:vDim];
    wTot=ones(1,vDim);
    rank=[];
    finished=0;
    X = get_x(dat);
    
    
    loop=0;
    while ((alg.find_min==0 & vDim>feat) | (alg.find_min==1 & finished==0))
        % reweight data
        xTemp = X(:,left).*(ones(size(X,1),1)*wTot(left));
        
        % alg.child.kerparam=wTot(left);
        
        datTemp = set_x(dat,xTemp);
        [r,alg.child]=train(alg.child,datTemp);
        if isempty(alg.child.alpha) | (sum(isnan(alg.child.alpha))==length(alg.child.alpha))
            break;
        end;
        %% the second argument of get_w is useful only when one_vs_rest
        %% is used
        
        absW=abs(get_w(alg.child,3));
        wTot(left) = wTot(left).*absW;
        
        %% the value 1e3 may be too small, to be checked!
        f=find(abs(wTot(left))<=max(abs(wTot))/1e3);
        remove=left(f);
        wTot(remove)=0;
        kept=find(abs(wTot(left))>0);
        left=left(kept);
        dat=get(dat,[],kept);     %% cut features that are zero
        
        rank=[remove rank];
        vDim=length(left);
        disp(['l0, feat=' num2str(vDim) ' - iteration ' num2str(loop) '...']);
        if norm((absW(absW~=0))-ones(size(absW(absW~=0)))) <norm(absW(absW~=0))*10^(-3) finished=1; end;
        
        loop=loop+1;
        if loop>20,
            finished=1;
        end;
        if vDim == 0
            finished = 1;
        end;
    end; %% while
    disp('l0 converged.');
    % now rank remaining features according to abs(absW)
    [val ind]=sort(-abs(wTot(left)));
    rank=[left(ind) rank];
    if feat==0
        feat=length(absW); %% choose minimum of l0 norm as features
    end;
    alg.feat=feat;
    alg.rank=rank;
    dat=get(dorig,[],alg.rank([1:feat]));  % perform the feature selection
end;

dat=set_name(dat,[get_name(dat) ' -> ' get_name(alg)]);

if alg.output_rank==1  %% output selected features, not label estimates
    res=dat;
else
    %    alg.child.ker='linear';
    [res,alg.child]=train(alg.child,dat);
end
retAlg=alg;


