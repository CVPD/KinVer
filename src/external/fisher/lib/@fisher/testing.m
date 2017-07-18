function results = testing(alg,dat)

 if alg.output_rank==1  %% output features with best correlation scores
   feat=alg.feat; 
   if isempty(feat) 
       feat=length(alg.rank);   
   end;
   results=get(dat,[],alg.rank(1:feat));     %% perform feature selection
 else
   X = get_x(dat);
   [numEx,vDim,oDim] = get_dim(dat);
   feat=alg.feat;
   if isempty(feat) 
       feat=size(X,2); 
   end;
   rnkToFeat=alg.rank(1:feat); 
   
   if size(alg.w,1)>1,
    %% Computation of b
    btmp=zeros(oDim,oDim);
    xTrain=get_x(alg.d);
    yTrain=get_y(alg.d);
    
    for i=1:oDim,
        for j = i+1:oDim,
            Posidx = find(yTrain(:,i)==1);
            Posjdx = find(yTrain(:,j)==1);
            milieu = (mean(xTrain(Posidx,rnkToFeat))+mean(xTrain(Posjdx,rnkToFeat)))/2;
            btmp(i,j)= -(alg.w(rnkToFeat,i)-alg.w(rnkToFeat,j))'*milieu';
            btmp(j,i)= -btmp(i,j);
        end;
    end;
       
    % compute the values of b_i s.t. sum_i b_i = 0
    bo = zeros(oDim,1); % the bias vector
    for i=1:oDim,
      temp = sum(btmp(i,:))/oDim;
      for j=1:oDim,
        bo(j) =bo(j)+temp-btmp(i,j);
      end;
    end;
    bo=bo/oDim;
    %% compute the output on the test set
    yEst = X(:,rnkToFeat)*alg.w(rnkToFeat,:) + ones(size(X,1),1)*bo';   
    [r,Ytmp2]=max(yEst,[],2);
    yEst = -ones(size(yEst));
    for i=1:length(Ytmp2),
        yEst(i,Ytmp2(i))=1;
    end;      
   else
   %% Two-class case
   yEst=(alg.w(rnkToFeat)*X(:,rnkToFeat)'+(-alg.w(rnkToFeat)*alg.b0(rnkToFeat)'))';
   if alg.algorithm.use_signed_output
     yEst=sign(yEst);
   end
   
   end;
   dat=set_x(dat,yEst);
   results=dat;
 end;
 results=set_name(results,[get_name(results) ' -> ' get_name(alg)]);
 
  
 
