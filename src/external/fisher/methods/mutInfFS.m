function [ rank , w] = mutInfFS( X,Y,numF )
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
% fprintf('\n+ Feature selection method: Mut-Inf \n');
rank = [];
for i = 1:size(X,2)
    rank = [rank; -muteinf(X(:,i),Y) i];
end;
rank = sortrows(rank,1);	
w = rank(1:numF, 1);
rank = rank(1:numF, 2);

end

