% function calculateSaveFeatures(imageDir,convnetDir,outputFileName)
% 
% Extracts the features from the images and saves them in a file.
% Each file contains all the features extracted for a specific kinship relationship
%
% Input: imageDir; Directory from where the current kinship relationship's images are loaded
% Input: convnetDir; directory where the deep networks are stored
% Input: outputFileName;  destiny file name that contains all the features for a kinship relationship
% Output: data; cell array containing the feature vectors
function [data] = calculateSaveFeatures(imageDir,convnetDir,outputFileName)

%%%%%%%%%%%%%%%% Initialisations %%%%%%%%%%%%%%%%

% Load the model and upgrade it to MatConvNet current version.
oldFolder = cd(convnetDir);
run matlab/vl_setupnn;
% Load vgg net
vggFace = load('vgg-face.mat');
vggFace = vl_simplenn_tidy(vggFace);
% Load vggF
vggF = load('vgg-f.mat');
vggF = vl_simplenn_tidy(vggF);
cd(oldFolder);

imageNames = dir(strcat(imageDir,'/*.jpg'));
numImages = length(imageNames);
disp( ['number of images: ' num2str(numImages)] );
disp( ['output file name: ' outputFileName] );
if exist(outputFileName, 'file') == 2 % If file exists load, else initialise
    load(outputFileName,'data');
else
    data = cell(numImages,1);
end

%%%%%%%%%%%%%%%% End of initialisations %%%%%%%%%%%%%%%%



% Save image names if do not exist
if isfield(data{1},'name') == 0
    for idx = 1:numImages
        data{idx}.name = imageNames(idx).name;
    end
end
% Save vgg features if do not exist
tic()
if isfield(data{1},'vggFaceFeat') == 0
    for idx = 1:numImages
        imageFullPath = strcat(imageDir,'/',data{idx}.name);
        % Pre-processing
        im = imread( imageFullPath );
        im_ = single(im);
        im_ = imresize(im_, vggFace.meta.normalization.imageSize(1:2));
        im_ = im_ - vggFace.meta.normalization.averageImage;
        % Network pass
        res = vl_simplenn(vggFace, im_);
        % Store features of the layer previous to the classification
        data{idx}.vggFaceFeat = gather(res(end-2).x);
    end
end

% Save vggF features if do not exist
if isfield(data{1},'vggFFeat') == 0
    for idx = 1:numImages
        imageFullPath = strcat(imageDir,'/',data{idx}.name);
        % Pre-processing
        im = imread( imageFullPath );
        im_ = single(im);
        im_ = imresize(im_, vggF.meta.normalization.imageSize(1:2));
        im_ = im_ - vggF.meta.normalization.averageImage;
        % Network pass
        res = vl_simplenn(vggF, im_);
        % Store features of the layer previous to the classification
        data{idx}.vggFFeat = gather(res(end-2).x);
    end
end
toc()

% Save LBP features if do not exist
if isfield(data{1},'LBPFeat') == 0
    for idx = 1:numImages
        imageFullPath = strcat(imageDir,'/',data{idx}.name);
        % Pre-processing
        im = imread( imageFullPath );
        gray = rgb2gray(im); % image to grey scale
        
        % Store LBP features
        data{idx}.LBPFeat = extractLBPFeatures(gray,'NumNeighbors', 16, 'Radius', 4);
    end
end

% Save HOG features if do not exist
if isfield(data{1},'HOGFeat') == 0
    for idx = 1:numImages
        imageFullPath = strcat(imageDir,'/',data{idx}.name);
        % Pre-processing
        im = imread( imageFullPath );
        gray = rgb2gray(im); % image to grey scale
        
        % Store HOG features
        data{idx}.HOGFeat = extractHOGFeatures(gray, 'CellSize', [15 15], 'NumBins', 13); % cell size of 19x19 and 12 bins
    end
end

save(outputFileName,'data');
disp( ['Save feature data as ' outputFileName] );
end