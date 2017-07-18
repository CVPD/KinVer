function [ rank , w] = fsvFS( X,Y,numF )
% Matlab Code-Library for Feature Selection
% Support: Giorgio Roffo email: giorgio.roffo@univr.it
%  If you use our toolbox please cite our paper:
% 
%  BibTex
%  ------------------------------------------------------------------------
%     @InProceedings{Roffo_2015_ICCV,
%     author = {Roffo, Giorgio and Melzi, Simone and Cristani, Marco},
%     title = {Infinite Feature Selection},
%     journal = {The IEEE International Conference on Computer Vision (ICCV)},
%     month = {June},
%     year = {2015}
%     }
%  ------------------------------------------------------------------------
fprintf('\n+ Feature selection method: FSV \n');
 
loop=0;
finished=0;
alpha = 5; % Default
[m,n] = size(X);
v = zeros(n,1);

%% Main LOOP
while (~finished),
    loop=loop+1;    
    scale = alpha*exp(-alpha*v);
    
    A=[diag(Y)*X, -diag(Y)*X, Y, -Y, -eye(m)];
    Obj = [scale',scale',0,0,zeros(1,m)];
    b = ones(m,1);
    x = slinearsolve(Obj',A,b,Inf);
    w = x(1:n)-x(n+1:2*n);
    b0 = x(2*n+1)-x(2*n+2);
    vnew=abs(w);
    
    if (norm(vnew-v,1)<10^(-5)*norm(v,1)),
        finished=1;
    else
        v=vnew;
    end;
    if (loop>20),
        finished=1;
    end;
    nfeat=length(find(vnew>100*eps));
    
    disp(['Iter ' num2str(loop) ' - feat ' num2str(nfeat)]);
    
    if nfeat<numF,
        finished=1;
    end;
end;

[~, ind] = sort(-abs(w));

w=w;
rank=ind;


end

