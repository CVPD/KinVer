function [res,retAlg] =  training(alg,d)

disp(['training ' get_name(alg) '.... '])
dorig=d;

[X Y]  = get_xy(d);
[numEx,vDim,oDim]=get_dim(d);
feat=alg.feat; % number of features desired
rank=[];

%% if we use the previous trainings
if alg.algorithm.use_prev_train==1&alg.algorithm.trained==1,
    d=get(d,[],alg.rank(1:alg.feat));
else

    if oDim<2,
        %% if two-class
        corr = (mean(X(Y==1,:))-mean(X(Y==-1,:))).^2;
        st   = std(X(Y==1,:)).^2;
        st   = st+std(X(Y==-1,:)).^2;
        f=find(st==0); %% remove ones where nothing occurs
        st(f)=10000;  %% remove ones where nothing occurs
        corr = corr ./ st;

        if alg.output_rank==1  %% output features with best correlation 

            f=find(abs(corr)>10000);
            corr(f)=0;
            [dummy alg.rank]=sort(-abs(corr));
            %[dummy alg.rank]=sort(-(corr)); %% why was this ever here?
            %%%%%
        else
            %%%%
            [dummy alg.rank]=sort(-abs(corr));
            %%%%
            alg.w=corr;
            posIdX=find(Y>0);
            Negidx=find(Y<0);
            alg.b0= (mean(X(posIdX,:))+mean(X(Negidx,:)))/2;
            %%%%% alg.rank=rank;
        end;%% if alg.output...

    else %% if oDim>=2...

        %% Compute the std for everybody
        Std = zeros(1,vDim);
        for i=1:oDim,
            indQ = find(Y(:,i)==1);
            Std = Std + std(X(indQ,:),1);
        end;
        for i=1:oDim,
            indp = find(Y(:,i)==1);
            Wtmp(:,i) = (mean(X(indp,:))./Std)';
        end;
        indTemp=[];
        corrTemp=[];
        for j=1:oDim,
            WW = [Wtmp(:,1:j-1),Wtmp(:,j+1:oDim)];
            WW = (Wtmp(:,j)*ones(1,oDim-1) - WW);
            switch alg.method,
                case 1,
                    rankW = min(WW,[],2);
                case 2,
                    rankW = sum(WW')';
                case 3,
                    r1 = min(WW,[],2);
                    r2 = max(WW,[],2);
                    rankW = (r1+r2)/2;
            end;

            [u,v] = sort(-abs(rankW'));
            indTemp = [indTemp,v'];
            corrTemp = [corrTemp, u'];
        end;
        clear WW;

        indTemp = reshape(indTemp',1,oDim*vDim);
        corrTemp = reshape(corrTemp',1,oDim*vDim);
        %%little trick to take the unique elements from
        %% indTemp without ordering them. (matlab orders
        %% the elements when using unique.)
        temp=fliplr(indTemp);
        corrTemp = fliplr(corrTemp);

        [u,v] = unique(temp);

        [w,s]=  sort(v);
        corrTemp = corrTemp(v);

        temp=fliplr(u(s));
        alg.rank=temp;

        alg.w = corrTemp;

        clear indTemp;

        %% compute b
        btmp = zeros(oDim,oDim);
        for i=1:oDim,
            for j = i+1:oDim,
                posIdX = find(Y(:,i)==1);
                posJdX = find(Y(:,j)==1);
                milieu = (mean(X(posIdX,:))+mean(X(posJdX,:)))/2;
                btmp(i,j)= -(Wtmp(:,i)-Wtmp(:,j))'*milieu';
                btmp(j,i)= -btmp(i,j);
            end;
        end;
        %% output
        alg.b0 = btmp;

        clear btmp;
        pack;
        alg.d=d;
        % alg.w=Wtmp;
    end;
end;
res=test(alg,dorig);
retAlg=alg;
