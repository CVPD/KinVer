%% KinFace-I and KinFace-II datasets
oldDir = cd('..');
mkdir datasets;
cd('datasets');
unzip('http://www.kinfacew.com/dataset/KinFaceW-I.zip')
unzip('http://www.kinfacew.com/dataset/KinFaceW-II.zip')
cd(oldDir);

%% External
oldDir = cd('external');
% MNRML code
unzip('http://www.kinfacew.com/codes/NRML.zip');
cd(oldDir);
message = 'Setup done. Only left to Download zip file from https://es.mathworks.com/matlabcentral/fileexchange/56937-feature-selection-library and extract it into a file called fisher inside the external folder';
msgbox(message)