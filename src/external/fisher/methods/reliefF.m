function [RANKED, WEIGHT] = reliefF( X, Y, K )
%   [RANKED,WEIGHT] = relieff(X,Y,K) computes ranks and weights of
%     attributes (predictors) for input data matrix X and response vector Y
%     using ReliefF algorithm for classification or RReliefF for regression
%     with K nearest neighbors. For classification, relieff uses K nearest
%     neighbors per class. RANKED are indices of columns in X ordered by
%     attribute importance, meaning RANKED(1) is the index of the most
%     important predictor. WEIGHT are attribute weights ranging from -1 to 1
%     with large positive weights assigned to important attributes.
%  
%     If Y is numeric, relieff by default performs RReliefF analysis for
%     regression. If Y is categorical, logical, a character array, or a cell
%     array of strings, relieff by default performs ReliefF analysis for
%     classification.
%  
%     Attribute ranks and weights computed by relieff usually depend on K. If
%     you set K to 1, the estimates computed by relieff can be unreliable for
%     noisy data. If you set K to a value comparable with the number of
%     observations (rows) in X, relieff can fail to find important
%     attributes. You can start with K=10 and investigate the stability and
%     reliability of relieff ranks and weights for various values of K.
%
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

fprintf('\n+ Feature selection method: Relief-F \n');
%% Wrapper: use Matlab implementation
[RANKED,WEIGHT] = relieff(X,Y,K);
% Matlab Code-Library for Feature Selection
% Contact: Giorgio Roffo email: giorgio.roffo@univr.it
