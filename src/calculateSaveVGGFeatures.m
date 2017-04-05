
function calculateSaveVGGFeatures(imageDir,convnetDir,outputFileName)

%%%%%%%%%%%%%%%% Initialisations %%%%%%%%%%%%%%%%

% Load the model and upgrade it to MatConvNet current version.
oldFolder = cd(convnetDir);
run matlab/vl_setupnn;
% Load vgg net
vggNet = load('vgg-face.mat');
vggNet = vl_simplenn_tidy(vggNet);
% Load ImageNet
imageNet = load('imagenet-vgg-f.mat');
imageNet = vl_simplenn_tidy(imageNet);
cd(oldFolder);

%%%%%%%%%%%%%%%% End of initialisations %%%%%%%%%%%%%%%%

imageNames = dir(strcat(imageDir,'/*.jpg'));
numImages = length(imageNames);

data = cell(numImages,1);

% Save image names
for idx = 1:numImages
    data{idx}.name = imageNames(idx).name;
end

% Save vgg features
for idx = 1:numImages
    imageFullPath = strcat(imageDir,'/',data{idx}.name);
    % Pre-processing
    im = imread( imageFullPath );
    im_ = single(im);
    im_ = imresize(im_, vggNet.meta.normalization.imageSize(1:2));
    im_ = im_ - vggNet.meta.normalization.averageImage;
    % Network pass
    res = vl_simplenn(vggNet, im_);
    % Store features of the layer previous to the classification
    data{idx}.vggFeat = gather(res(end-2).x);
end

% Save ImageNet features
for idx = 1:numImages
    imageFullPath = strcat(imageDir,'/',data{idx}.name);
    % Pre-processing
    im = imread( imageFullPath );
    im_ = single(im);
    im_ = imresize(im_, imageNet.meta.normalization.imageSize(1:2));
    im_ = im_ - imageNet.meta.normalization.averageImage;
    % Network pass
    res = vl_simplenn(imageNet, im_);
    % Store features of the layer previous to the classification
    data{idx}.imagenetFeat = gather(res(end-2).x);
end

save(outputFileName,'data');
end