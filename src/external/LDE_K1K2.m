% Implementation of the LDE presented in the following paper: 
    % Chen, H.T., Chang, H.W., Liu, T.L.: Local discriminant embedding and its variants.
    %In: Computer Vision and Pattern Recognition, 2005. CVPR 2005. IEEE Computer
    %Society Conference on. Volume 2., IEEE (2005) 846–853
% I do not own the code
function [vec val Ww Wb Lw Lb] = LDE_K1K2(X, labels , K1, K2 , beta )
%% Local Descriminant Embedding with fix number of neighbours for homogeneous and heterogeneus samples
% 
% Input
%       X       ==   input data, a matrix, each input should be a column vector
%       labels  ==   column vector, containing the labels of inputs
%       K1      ==   number of homogeneous neighbours
%       K2      ==   number of heterogeneous neighbours
%       beta    ==   regularized value for LDE criteria, 
%                         when there is SSS problem and it is not defined, will be set to 0.001
% Output
%       vec     ==   eigenvectors calculated 
%       val     ==   corresponding eigenvalues
%       Ww      ==  Weight matrix, homogeneous samples
%       Wb      ==  Weight matrix, heterogeneous sample
%                         
%                         
% %Sample code
% applying LDE on 15 different classes and 10 inputs per class.
%     X = rand(100,150);
%     labels = repmat(1:15,1,10);
%     K1 = 4;
%     K2 = 6;
% 
%     [vec val Ww Wb Lb Lw] = LDE_K1K2(X,labels,K1,K2);
%     X_prj = vec' * X;
%     scatter3(X_prj(1,:),X_prj(2,:),X_prj(3,:),80,labels)

%% Input checking
if nargin < 4
    error('The number of inputs should be at least 4')
end
if nargin ==4
    beta = 0;
end

[feadim nSmp] = size(X);
if feadim > nSmp && beta == 0
    beta = 0.001;
end

if nSmp ~= length(labels)
    error('Number of samples and labels do not match');
end


 %% Calculating Similarity Matrix

    aa = sum(X .* X);
    ab = X' * X;
    M_distance =      repmat(aa',1,nSmp) + repmat(aa, nSmp,1) - 2*ab';
    M_distance (abs(M_distance )<1e-10)=0;
    sigma2       =   mean( (M_distance(:)));
    MatSim      =   exp( -(M_distance) / (2*sigma2) );
clear sigma aa ab
%% The neighborcity Criteria

    MatCritBin=zeros(size(labels,2));
    [tmp, idx] =    sort(M_distance,2);
    
    for h=1:size(idx,1)
        Dis_h= idx(h,:);
        Slbl = labels(Dis_h);
        Homo =  find( Slbl == labels(h) );
        Htro =  find( Slbl ~= labels(h) );
        MatCritBin(h,Dis_h(Homo(1:K1)))=1;
        MatCritBin(h,Dis_h(Htro(1:K2)))=1;
    end
    MatCritBin  =   or(MatCritBin,MatCritBin');%To assure the symmetricity
            
clear MatMeanSim MatCriteria M_distance Dis_h Slbl h 
   
 %% Buiding Wb and Ww
    Wb = ones (size(MatCritBin));
    Ww = zeros(size(MatCritBin));

    %%The HomoG. HetroG. Criteria
    label=unique(labels);
    for i=1:size(label,2)
        found   =   find(labels == label(i)); % find Homo points
        Wb(found,found)=  0; %Change those points as 0 in Between Matrix
        Ww(found,found)=  1; %Change those points as 1 in Within  Matrix 
    end
    
    %%A point is not considered as its neighbour
    for i=1:size(Ww,1)
        Ww(i,i)=0;
    end

    Wb=MatCritBin.*Wb;%MatSim.*
    Ww=MatCritBin.*MatSim.*Ww;

%% Distriminant Criteria

Lb  =    full( sparse(1:size(Wb,1) , 1:size(Wb,2) , sum(Wb) , size(Wb,1) , size(Wb,2)) ) - Wb;
Lw  =    full( sparse(1:size(Ww,1) , 1:size(Ww,2) , sum(Ww) , size(Ww,1) , size(Ww,2)) ) - Ww;

    Stb = X* Lb* X';
    Stw = X* Lw* X';

%% LDE Criterias 

    Stw     =   Stw + beta * trace(Stw) * eye(size(Stw));
    Stw = Stw./max(Stw(:));
    Stb = Stb./max(Stb(:));
    t   =  Stw^-1 * Stb  ;
    [vec val] = eig( t ./ max(t(:)) );


val = real(diag(val));
[val idx] = sort( val , 'descend' );
vec = real(vec(:,idx));
