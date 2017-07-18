% File that must be run before the first execution of the main in order to prepare the datasets and libraries.
% Also the Feature Selection Library must be installed using the link provided by the window created at the execution.

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
message = 'Setup done. Only left to add the Matlab's feature selection library that performs fisher feature selection. To install it, first download the zip file from this url https://es.mathworks.com/matlabcentral/fileexchange/56937-feature-selection-library and then extract its content into a new folder called "fisher" created into the path src/external';
msgbox(message)