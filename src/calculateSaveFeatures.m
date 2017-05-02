
function calculateSaveFeatures(imageDir,convnetDir,outputFileName)

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
    im_ = imresize(im_, vggFace.meta.normalization.imageSize(1:2));
    im_ = im_ - vggFace.meta.normalization.averageImage;
    % Network pass
    res = vl_simplenn(vggFace, im_);
    % Store features of the layer previous to the classification
    data{idx}.vggFeat = gather(res(end-2).x);
end

% Save vggF features
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

save(outputFileName,'data');
end